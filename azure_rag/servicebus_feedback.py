"""
servicebus_feedback.py — PHASE 12b: Feedback Async Queue (Azure Service Bus).

The Streamlit HITL form sends the clinician's decision here instead of
writing directly to the audit store — decoupling the UI from downstream
consumers (audit trail writer, RAGAS eval sampler, prescription generator).
A dead-letter queue catches anything a consumer can't process after retries,
so a malformed feedback event can never silently vanish.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from azure.identity import DefaultAzureCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.exceptions import ServiceBusError

SERVICEBUS_NAMESPACE = os.environ.get("AZURE_SERVICEBUS_NAMESPACE")  # "<ns>.servicebus.windows.net"
FEEDBACK_QUEUE_NAME = os.environ.get("AZURE_SERVICEBUS_FEEDBACK_QUEUE", "hitl-feedback")
CONNECTION_STRING = os.environ.get("AZURE_SERVICEBUS_CONNECTION_STRING")  # omit to use Managed Identity


def _client() -> Optional[ServiceBusClient]:
    if CONNECTION_STRING:
        return ServiceBusClient.from_connection_string(CONNECTION_STRING)
    if SERVICEBUS_NAMESPACE:
        return ServiceBusClient(fully_qualified_namespace=SERVICEBUS_NAMESPACE, credential=DefaultAzureCredential())
    return None


def send_hitl_feedback(event: Dict[str, Any]) -> bool:
    """
    event should already be a fully-formed audit-style dict (see audit.py's
    make_audit_event) — this function just gets it onto the queue reliably.
    Returns False (never raises) if Service Bus isn't configured, so local
    dev/demo mode degrades to "feedback not queued" rather than crashing
    the HITL form submit.
    """
    client = _client()
    if client is None:
        return False

    try:
        with client:
            sender = client.get_queue_sender(queue_name=FEEDBACK_QUEUE_NAME)
            with sender:
                message = ServiceBusMessage(json.dumps(event, default=str))
                message.content_type = "application/json"
                message.correlation_id = event.get("event_id")
                sender.send_messages(message)
        return True
    except ServiceBusError:
        return False


def receive_and_process_feedback(handler) -> None:
    """
    Run this inside an Azure Function with a Service Bus trigger (preferred
    for production) or as a standalone worker loop for local dev. `handler`
    is a callable(event: dict) -> None that does the actual work (write to
    audit store, trigger prescription generation, sample for RAGAS eval).
    Messages that repeatedly fail `handler` land in the queue's dead-letter
    sub-queue automatically after the configured max delivery count.
    """
    client = _client()
    if client is None:
        raise RuntimeError("Service Bus is not configured (set AZURE_SERVICEBUS_NAMESPACE or "
                            "AZURE_SERVICEBUS_CONNECTION_STRING).")

    with client:
        receiver = client.get_queue_receiver(queue_name=FEEDBACK_QUEUE_NAME, max_wait_time=30)
        with receiver:
            for msg in receiver:
                try:
                    event = json.loads(str(msg))
                    handler(event)
                    receiver.complete_message(msg)
                except Exception:
                    # Let Service Bus's built-in max-delivery-count move this to DLQ
                    # rather than abandon-looping forever on a poison message.
                    receiver.abandon_message(msg)
