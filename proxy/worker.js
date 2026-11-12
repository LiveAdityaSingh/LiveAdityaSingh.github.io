// Cloudflare Worker: chat proxy for the portfolio chatbot.
// Keeps the Groq API key and the system prompt off the public site.
//
// Deploy (from this folder):
//   npx wrangler login
//   npx wrangler secret put GROQ_API_KEY
//   npx wrangler deploy
// Then set REACT_APP_CHAT_ENDPOINT to the worker URL in the site's .env and rebuild.
import { PORTFOLIO_CONTEXT } from '../src/chatContext.js';

const ALLOWED_ORIGINS = ['https://liveadityasingh.github.io', 'http://localhost:3000'];
const MAX_MESSAGES = 12;
const MAX_MESSAGE_CHARS = 1000;
const MAX_TOKENS = 500;

// Per-IP rate limit. Held in memory per worker instance, so it is best-effort.
const RATE_WINDOW_MS = 60 * 1000;
const RATE_MAX_REQUESTS = 10;
const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < RATE_WINDOW_MS);
  recent.push(now);
  hits.set(ip, recent);
  return recent.length > RATE_MAX_REQUESTS;
}

function json(body, status, cors) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...cors },
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin');
    if (!ALLOWED_ORIGINS.includes(origin)) {
      return json({ error: 'Origin not allowed' }, 403, {});
    }
    const cors = {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Vary': 'Origin',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }
    if (request.method !== 'POST') {
      return json({ error: 'Method not allowed' }, 405, cors);
    }
    if (rateLimited(request.headers.get('CF-Connecting-IP') || 'unknown')) {
      return json({ error: 'Too many requests' }, 429, cors);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: 'Invalid JSON' }, 400, cors);
    }
    if (!Array.isArray(body.messages)) {
      return json({ error: 'messages must be an array' }, 400, cors);
    }

    // Only user/assistant turns are accepted; the system prompt is added here
    const history = body.messages
      .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .slice(-MAX_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_MESSAGE_CHARS) }));
    if (history.length === 0) {
      return json({ error: 'No messages' }, 400, cors);
    }

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'openai/gpt-oss-20b',
        messages: [{ role: 'system', content: PORTFOLIO_CONTEXT }, ...history],
        temperature: 0.3,
        // gpt-oss is a reasoning model: keep reasoning short so it fits the token cap
        reasoning_effort: 'low',
        max_tokens: MAX_TOKENS,
      }),
    });
    if (!response.ok) {
      return json({ error: 'Upstream error', upstreamStatus: response.status }, 502, cors);
    }

    const data = await response.json();
    return json({ reply: data.choices[0].message.content }, 200, cors);
  },
};
