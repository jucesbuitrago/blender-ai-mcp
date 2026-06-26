-- Script-memory (RAG) schema for the blender_scripts table.
-- Run once in the Supabase SQL editor (or via `supabase db` / psql).
-- Assumes blender_scripts(id, prompt_original, script_python, notas) already exists.
-- Embedding dim 384 matches all-MiniLM-L6-v2 (see server/infrastructure/embeddings.py).

-- 1. Enable pgvector.
create extension if not exists vector;

-- 2. Add the embedding column.
alter table blender_scripts
  add column if not exists embedding vector(384);

-- 3. Approximate-nearest-neighbour index for cosine distance.
--    ivfflat needs ANALYZE/rows to be effective; fine for small tables too.
create index if not exists blender_scripts_embedding_idx
  on blender_scripts using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Similarity-search RPC used by buscar_referencias.
--    NOTE: `id bigint` assumes a bigint/identity PK. If your id is uuid,
--    change the return column type to uuid.
create or replace function match_blender_scripts(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id bigint,
  prompt_original text,
  script_python text,
  notas text,
  similarity float
)
language sql
stable
as $$
  select
    blender_scripts.id,
    blender_scripts.prompt_original,
    blender_scripts.script_python,
    blender_scripts.notas,
    1 - (blender_scripts.embedding <=> query_embedding) as similarity
  from blender_scripts
  where blender_scripts.embedding is not null
  order by blender_scripts.embedding <=> query_embedding
  limit match_count;
$$;
