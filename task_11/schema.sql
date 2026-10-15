-- 1. Create threads table to store conversation sessions
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create messages table to store chat history per thread
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create index for fast message retrieval by thread
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id, created_at ASC);