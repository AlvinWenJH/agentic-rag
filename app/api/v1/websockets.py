from typing import Optional, List
import json
import asyncio
import structlog
from fastapi import APIRouter, WebSocket, Query
from fastapi.websockets import WebSocketDisconnect

from app.core.cache import get_cache_client


logger = structlog.get_logger()
router = APIRouter()


@router.websocket("/status")
async def status_updates(
    websocket: WebSocket,
    channel: Optional[str] = Query(None),
    channels: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
):
    await websocket.accept()

    try:
        client = get_cache_client()
        pubsub = client.pubsub()

        subs: List[str] = []
        if channels:
            subs.extend([c.strip() for c in channels.split(",") if c.strip()])
        if channel:
            subs.append(channel)
        if resource_type and resource_id:
            subs.append(f"status:{resource_type}:{resource_id}")
        if not subs:
            subs.append("status:global")

        await pubsub.subscribe(*subs)

        await websocket.send_json({"type": "subscribed", "channels": subs})

        async def reader():
            try:
                async for message in pubsub.listen():
                    if message and message.get("type") == "message":
                        data = message.get("data")
                        try:
                            payload = json.loads(data)
                        except Exception:
                            payload = {"type": "raw", "data": data}
                        try:
                            logger.info("Sending message : " + json.dumps(payload))
                        except Exception:
                            pass
                        await websocket.send_json(payload)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(reader())

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await pubsub.unsubscribe(*subs)
            await pubsub.close()

    except Exception as e:
        logger.error("WebSocket status handler error", error=str(e))
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            return
