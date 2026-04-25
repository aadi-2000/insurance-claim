"""
SSE (Server-Sent Events) Manager for real-time agent status updates.
"""
import asyncio
from typing import Dict, Set
import json
import logging

logger = logging.getLogger("insurance_claim_ai.sse")


class SSEManager:
    """Manages SSE connections and broadcasts agent status updates."""
    
    def __init__(self):
        self.active_streams: Dict[str, Set[asyncio.Queue]] = {}
        
    def create_stream(self, claim_id: str) -> asyncio.Queue:
        """Create a new SSE stream for a claim."""
        
        if claim_id not in self.active_streams:
            self.active_streams[claim_id] = set()
        
        queue = asyncio.Queue()
        self.active_streams[claim_id].add(queue)
        
        total_streams = len(self.active_streams[claim_id])
        
        logger.info(f"Created SSE stream for claim {claim_id}")
        return queue
    
    def remove_stream(self, claim_id: str, queue: asyncio.Queue):
        """Remove an SSE stream."""
        if claim_id in self.active_streams:
            self.active_streams[claim_id].discard(queue)
            if not self.active_streams[claim_id]:
                del self.active_streams[claim_id]
                logger.info(f"Removed all SSE streams for claim {claim_id}")
    
    async def send_event(self, claim_id: str, event_type: str, data: Dict):
        """Send an event to all streams for a claim."""
        
        if claim_id not in self.active_streams:
            return
        
        event_data = {
            "type": event_type,
            "data": data
        }
        
        num_streams = len(self.active_streams[claim_id])
        logger.info(f"Broadcasting {event_type} event for claim {claim_id}: {data}")
        
        # Send to all active streams for this claim
        for idx, queue in enumerate(self.active_streams[claim_id], 1):
            try:
                await queue.put(event_data)
            except Exception as e:
                logger.error(f"Failed to send event to queue: {e}")
        
    
    async def send_agent_status(self, claim_id: str, agent_name: str, status: str, confidence: float = None):
        """Send agent status update."""
        
        data = {
            "agent": agent_name,
            "status": status,
        }
        if confidence is not None:
            data["confidence"] = confidence
        
        await self.send_event(claim_id, "agent_status", data)
    
    async def send_completion(self, claim_id: str, result: Dict):
        """Send final completion event."""
        await self.send_event(claim_id, "complete", result)


# Global SSE manager instance
sse_manager = SSEManager()
