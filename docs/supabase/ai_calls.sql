-- MoontechPro / Lovable: ai_calls table for voice server direct writes.
-- Voice server upserts on call start (in_progress) and call end (completed + transcript).
-- Hangup webhooks merge recording_url, outcome, duration (provider = exotel | vobiz).

create table if not exists public.ai_calls (
  id uuid primary key default gen_random_uuid(),
  call_sid text not null,
  user_id uuid references auth.users (id) on delete set null,
  agent_id uuid,
  phone_number_id uuid,
  provider text not null default 'exotel',
  direction text not null default 'inbound',
  from_number text,
  to_number text,
  status text not null default 'in_progress',
  outcome text,
  started_at timestamptz,
  ended_at timestamptz,
  duration_seconds integer,
  recording_url text,
  transcript jsonb default '[]'::jsonb,
  channel_reservation_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint ai_calls_call_sid_key unique (call_sid)
);

create index if not exists ai_calls_user_started_idx
  on public.ai_calls (user_id, started_at desc);

create index if not exists ai_calls_agent_started_idx
  on public.ai_calls (agent_id, started_at desc);

create index if not exists ai_calls_provider_idx
  on public.ai_calls (provider, started_at desc);

-- Service role used by voice server bypasses RLS. If using anon key, add policies for user read.

comment on column public.ai_calls.provider is 'Telephony provider: exotel | vobiz';
comment on column public.ai_calls.phone_number_id is 'FK to phone_numbers.id when inbound mapped';
