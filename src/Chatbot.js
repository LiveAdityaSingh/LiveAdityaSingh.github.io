import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Bot } from 'lucide-react';
import './Chatbot.css';
import { PORTFOLIO_CONTEXT } from './chatContext';

// Chat proxy URL (see proxy/worker.js). The Groq key lives there, not in this bundle.
const CHAT_ENDPOINT = process.env.REACT_APP_CHAT_ENDPOINT;
const OFFLINE_MESSAGE = 'Chat is offline right now. Please email adityasingh2897@gmail.com instead.';

async function requestReply(history) {
  if (CHAT_ENDPOINT) {
    const response = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history })
    });
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    const data = await response.json();
    return data.reply;
  }

  // Legacy path: calls Groq straight from the browser, which exposes the key.
  // Remove once REACT_APP_CHAT_ENDPOINT is set and the key has been rotated.
  const p1 = ['g', 's', 'k', '_'].join('');
  const suffix = process.env.REACT_APP_GROQ_API_KEY_SUFFIX;
  if (!suffix) {
    throw new Error('Chat is not configured');
  }

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${p1 + suffix}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'llama-3.1-8b-instant',
      messages: [{ role: 'system', content: PORTFOLIO_CONTEXT }, ...history],
      temperature: 0.3,
      max_tokens: 300,
    })
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  const data = await response.json();
  return data.choices[0].message.content;
}

export default function Chatbot({ isOpen, setIsOpen }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! I'm Aditya's AI assistant. Ask me anything about his experience, projects, or skills!" }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const toggleRef = useRef(null);
  const wasOpenRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Move focus into the dialog on open and back to the toggle on close
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    } else if (wasOpenRef.current) {
      toggleRef.current?.focus();
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, setIsOpen]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      // Error bubbles are UI-only, so they are left out of the history
      const history = [...messages.filter(m => m.role !== 'error'), userMessage]
        .map(m => ({ role: m.role, content: m.content }));
      const botReply = await requestReply(history);

      setMessages((prev) => [...prev, { role: 'assistant', content: botReply }]);
    } catch (error) {
      console.error("Chatbot error:", error);
      setMessages((prev) => [...prev, { role: 'error', content: OFFLINE_MESSAGE }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className={`chatbot-wrapper ${isOpen ? 'chatbot-wrapper--open' : ''}`}>
      {isOpen && (
        <div className="chatbot-window" role="dialog" aria-modal="true" aria-label="Chat with Aditya's assistant">
          <div className="chatbot-header">
            <div className="chatbot-header-title">
              <Bot size={20} strokeWidth={1.5} aria-hidden="true" />
              Aditya's Assistant
            </div>
            <button className="chatbot-header-close" onClick={() => setIsOpen(false)} aria-label="Close chat">
              <X size={20} strokeWidth={1.5} />
            </button>
          </div>

          <div className="chatbot-messages" aria-live="polite">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chatbot-message ${msg.role === 'user' ? 'user' : msg.role === 'error' ? 'error' : 'bot'}`}>
                {msg.content}
              </div>
            ))}
            {isTyping && (
              <div className="chatbot-typing" aria-label="Assistant is typing">
                <div className="chatbot-typing-dot"></div>
                <div className="chatbot-typing-dot"></div>
                <div className="chatbot-typing-dot"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} className="chatbot-input-form">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Aditya..."
              aria-label="Your question"
              className="chatbot-input"
            />
            <button type="submit" className="chatbot-submit-btn" disabled={isTyping || !input.trim()} aria-label="Send message">
              <Send size={18} strokeWidth={1.5} />
            </button>
          </form>
        </div>
      )}

      {!isOpen && (
        <button ref={toggleRef} className="chatbot-toggle-btn" onClick={() => setIsOpen(true)} aria-label="Open chat">
          <MessageSquare size={24} strokeWidth={1.5} />
        </button>
      )}
    </div>
  );
}
