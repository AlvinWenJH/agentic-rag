## Overview
- Build a chat interface on `/query` with the input docked at the bottom.
- Provide a mode selector: "Document" (implemented) and "Collection" (disabled, coming soon).
- Stream answers from the backend `POST /api/v1/query/document/{document_id}` and render tokens live.

## Components
- Create `frontend/src/components/query/chat-pane.tsx` (client) to host the interactive UI.
- Reuse existing shadcn components found under `frontend/src/components/ui`: `Tabs`, `DropdownMenu`, `Button`, `Input`, `Card`, `Separator`, `Sidebar` primitives.
- Keep `frontend/src/app/query/page.tsx` as a server component and mount the `ChatPane` within the content area.

## UI Structure
- Header: keep existing sidebar header in `frontend/src/app/query/page.tsx:1-48` and insert the chat pane in the main content.
- Mode selector: `Tabs` with `value="document" | "collection"` (collection trigger is disabled and labeled "Coming soon").
- Document picker: `DropdownMenu` listing documents fetched from the backend; shows selected title and stores `document_id`.
- Messages area: `Card` with `overflow-y-auto`, grows to fill space; shows user and assistant messages with simple bubbles.
- Bottom input bar: sticky within the inset (`sticky bottom-0`), contains `Input` and `Send` `Button`; submit on Enter.

## Data & Streaming
- Fetch backend base URL via `getBackendUrl()` from `@/lib/env`.
- Load documents with `GET ${backendUrl}/api/v1/documents/?skip=0&limit=20` (same shape used in `documents-dashboard.tsx`).
- When user submits:
  - Push a `user` message to history.
  - Start a streaming POST to `${backendUrl}/api/v1/query/document/${documentId}` with JSON body `{ query, user_id?: string }`.
  - Read `ReadableStream` and parse SSE lines prefixed with `data:`; handle event types from the backend:
    - `start` → show metadata (optional header line in assistant message)
    - `text_delta` → append content to the current assistant message
    - `tool_call` → inline small status line (e.g., "Calling get_subtree_by_paths")
    - `final_result` → finalize message; persist usage and references in state
    - `error` → show an error toast and finalize the assistant message with error state

## State Management
- Local `useState` hooks inside `ChatPane`:
  - `mode`: `"document" | "collection"` (default `document`)
  - `selectedDocument`: `{ id: string, title: string } | null`
  - `messages`: `{ role: "user" | "assistant" | "system"; content: string }[]`
  - `input`: string; `isStreaming`: boolean
- Guard: disable `Send` until a document is selected and input is non-empty; cancel streaming on tab change or new send.

## Styling & Layout
- Replace the placeholder `div` in `frontend/src/app/query/page.tsx` with `<ChatPane />`; preserve container classes (`flex flex-1 flex-col gap-4 p-4`).
- Messages area minimum height `min-h-[40vh]` to match current placeholder; grows with `flex-1`.
- Bottom bar uses `sticky bottom-0` + muted background + `border-t` for visual separation.

## Error & Empty States
- Documents fetch failure: show `DropdownMenu` trigger with an error badge and a retry option.
- No documents: show an inline message and keep the send bar disabled.
- Streaming errors: append error message to chat and surface a toast.

## Verification
- Manual: open `/query`, select a document, send a question, observe streaming text and finalization.
- Functional checks:
  - Input sticks to bottom and remains visible on scroll.
  - Disabled collection tab is visible and non-interactive.
  - Messages list scrolls and new content appears at the bottom.
  - Network calls use `getBackendUrl()` consistently.

## Files to Update
- `frontend/src/app/query/page.tsx` — import and render `ChatPane` within the content container.
- `frontend/src/components/query/chat-pane.tsx` — new client component implementing the described UI.

## Notes
- No new external libraries; use existing UI primitives and Tailwind classes.
- Keep secrets out of logs; rely on backend for model execution.
- Future: implement "Collection" mode using a similar streaming endpoint when available.