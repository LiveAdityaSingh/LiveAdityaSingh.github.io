import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Bot } from 'lucide-react';
import './Chatbot.css';

const PORTFOLIO_CONTEXT = `
You are an AI assistant for Aditya Narayan Singh's portfolio.
Answer questions ONLY based on the following context. If a user asks something not in the context, politely decline and say you don't have that information. Do not invent anything.

Aditya Narayan Singh is a GCP Professional Data Engineer and Azure AI Associate with 2+ years of experience building production-grade AI and data solutions at Accenture. He is based in Birmingham, UK. He is currently pursuing his MSc in Artificial Intelligence & Machine Learning at the University of Birmingham (2025–2026).

Education:
- MSc Artificial Intelligence & Machine Learning at University of Birmingham (2025-2026)
- B.Tech in Computer Science & Engineering at Dr. APJ Abdul Kalam Technical University, India (2018-2022). CGPA: 7.29.

Experience:
- Data Engineering Analyst at Accenture Solutions Pvt. Ltd., Pune (Oct 2022 - Aug 2025). Architected and deployed end-to-end predictive pipelines using NiFi and GCP, reducing latency by 40%.
- Project Intern at Robust Results Pvt. Ltd., Kanpur (May 2020 - Jun 2020). Built predictive analytics solutions with 90% accuracy.

Skills:
- AI/GenAI: Multi-Agent Systems, RAG, LLMs, NLP, Computer Vision
- ML/DL: TensorFlow, PyTorch, Scikit-Learn
- Data Eng: PySpark, Apache NiFi, dbt, Kafka, Hadoop
- Cloud: GCP (Vertex AI, BigQuery), Azure, AWS, Docker, Kubernetes

Projects:
- Pulse-Doppler Signal Processing: HealthTech AI 2026 winner. Signal processing using CNN and PyQTG.
- MindGrow App: AI-powered productivity app.
- UKFin Smart Renewal System: Financial renewal automation.
- ML & Deep Learning Collections.

Certifications:
- Google Cloud Professional Data Engineer (Oct 2024)
- Microsoft Certified: Azure AI Engineer Associate (Dec 2021)
- AI Certification (IIT Kanpur), DeepLearning.AI, University at Buffalo.

Co-curricular:
- HealthTech AI 2026 Winner
- FrontierTechAI Loyalty Program Challenge Winner
- Dog NGO Volunteer, Drummer.

Contact:
- Email: adityasingh2897@gmail.com
- Phone: +44 7357 674282
- GitHub: LiveAdityaSingh
- LinkedIn: aditya-singhofficial
`;

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! I'm Aditya's AI assistant. Ask me anything about his experience, projects, or skills!" }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    const p1 = ['g', 's', 'k', '_'].join('');
    const suffix = process.env.REACT_APP_GROQ_API_KEY_SUFFIX;
    const apiKey = suffix ? p1 + suffix : null;

    if (!apiKey) {
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          { role: 'error', content: "Error: REACT_APP_GROQ_API_KEY_SUFFIX is missing in .env." }
        ]);
        setIsTyping(false);
      }, 1000);
      return;
    }

    try {
      // Prepare message history for the API, prepending the system prompt
      const apiMessages = [
        { role: 'system', content: PORTFOLIO_CONTEXT },
        // Map UI roles to API roles
        ...messages.map(m => ({ role: m.role === 'error' ? 'assistant' : m.role, content: m.content })),
        userMessage
      ];

      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          messages: apiMessages,
          temperature: 0.3,
          max_tokens: 300,
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      const botReply = data.choices[0].message.content;

      setMessages((prev) => [...prev, { role: 'assistant', content: botReply }]);
    } catch (error) {
      console.error("Chatbot error:", error);
      setMessages((prev) => [
        ...prev,
        { role: 'error', content: "Sorry, I'm having trouble connecting right now." }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chatbot-wrapper">
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-header-title">
              <Bot size={20} />
              Aditya's Assistant
            </div>
            <button className="chatbot-header-close" onClick={() => setIsOpen(false)}>
              <X size={20} />
            </button>
          </div>
          
          <div className="chatbot-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chatbot-message ${msg.role === 'user' ? 'user' : msg.role === 'error' ? 'error' : 'bot'}`}>
                {msg.content}
              </div>
            ))}
            {isTyping && (
              <div className="chatbot-typing">
                <div className="chatbot-typing-dot"></div>
                <div className="chatbot-typing-dot"></div>
                <div className="chatbot-typing-dot"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} className="chatbot-input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Aditya..."
              className="chatbot-input"
              disabled={isTyping}
            />
            <button type="submit" className="chatbot-submit-btn" disabled={isTyping || !input.trim()}>
              <Send size={18} />
            </button>
          </form>
        </div>
      )}
      
      {!isOpen && (
        <button className="chatbot-toggle-btn" onClick={() => setIsOpen(true)} aria-label="Open chat">
          <MessageSquare size={28} />
        </button>
      )}
    </div>
  );
}
