import React, { useState, useEffect, Suspense } from 'react';
import { motion, MotionConfig } from 'framer-motion';
import { Menu, X, ArrowRight, ArrowUpRight, Github, Linkedin, Mail, Phone, FileDown } from 'lucide-react';
import './index.css';
import Chatbot from './Chatbot';

// Three.js is heavy, so the sphere loads in its own chunk after first paint
const AtomicBackground = React.lazy(() => import('./Sphere'));

/* ═══════════════════════════════════════════════════════════
   FRAMER MOTION VARIANTS
═══════════════════════════════════════════════════════════ */
const EASE_OUT = [0.23, 1, 0.32, 1];

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, delay, ease: EASE_OUT },
  }),
};

/* ═══════════════════════════════════════════════════════════
   REVEAL WRAPPER — section headings and key blocks only
═══════════════════════════════════════════════════════════ */
function Reveal({ children, className }) {
  return (
    <motion.div
      className={className}
      variants={fadeUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-10%' }}
    >
      {children}
    </motion.div>
  );
}

function Section({ id, title, children }) {
  return (
    <section id={id} className="section">
      <Reveal>
        <h2 className="section-title">{title}</h2>
      </Reveal>
      {children}
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════
   DATA
═══════════════════════════════════════════════════════════ */

const CV_URL = process.env.PUBLIC_URL + '/Aditya-Narayan-Singh-CV.pdf';

const NAV_LINKS = [
  { href: '#about', label: 'About' },
  { href: '#experience', label: 'Experience' },
  { href: '#projects', label: 'Projects' },
  { href: '#skills', label: 'Skills' },
  { href: '#contact', label: 'Contact' },
];

const SECTION_IDS = ['about', 'experience', 'projects', 'skills', 'education', 'publications', 'cocurricular', 'contact'];

const ABOUT_FACTS = [
  { label: 'Location', value: 'Birmingham, UK' },
  { label: 'Studying', value: 'MSc Artificial Intelligence & Machine Learning, University of Birmingham (2025–2026)' },
  { label: 'Previously', value: 'Data Engineering Analyst at Accenture (Oct 2022 – Aug 2025)' },
  { label: 'Awards', value: 'Winner of HealthTech AI 2026 and the FrontierTechAI Loyalty Program Challenge' },
  { label: 'Outside work', value: 'Drummer, volunteer dog carer and badminton enthusiast' },
];

const SKILLS = {
  'AI / GenAI': ['Multi-Agent Systems (ADK)', 'RAG', 'LLMs', 'NLP', 'Computer Vision', 'Prompt Engineering'],
  'ML / DL': ['TensorFlow', 'PyTorch', 'Scikit-Learn', 'Keras', 'XGBoost', 'OpenCV'],
  'Data Eng.': ['PySpark', 'Apache NiFi', 'dbt', 'Apache Kafka', 'Hadoop', 'Airflow', 'Informatica'],
  'Cloud': ['GCP (Vertex AI, BigQuery, Dataproc)', 'Azure', 'AWS', 'Docker', 'Kubernetes', 'GitLab CI'],
  'Languages': ['Python', 'SQL', 'JavaScript', 'R', 'Bash'],
  'Viz & BI': ['Tableau', 'Power BI', 'Matplotlib', 'Seaborn'],
};

const PROJECTS = [
  {
    name: 'Pulse-Doppler Signal Processing',
    award: 'Winner, HealthTech AI 2026',
    desc: 'An optimized solution for doppler signal processing and denoising, built with a CNN, frame slicing, a sliding window algorithm and PyQTG.',
    tags: ['Python', 'CNN', 'PyQTG', 'Signal Processing'],
    link: 'https://github.com/LiveAdityaSingh/Pulse-Doppler-Signal-Processing',
    featured: true,
  },
  {
    name: 'MindGrow App',
    desc: 'An AI-powered productivity and growth application built with JavaScript, designed to help users track and develop cognitive habits.',
    tags: ['JavaScript', 'React', 'AI', 'Node.js'],
    link: 'https://github.com/LiveAdityaSingh/MindGrow-app',
  },
  {
    name: 'ML Projects Collection',
    desc: 'A collection of machine learning projects: Mask Detection, Pose Estimation for workouts, Sentiment Analysis and Toxic Comments Recognition models.',
    tags: ['Python', 'Scikit-Learn', 'TensorFlow', 'Jupyter'],
    link: 'https://github.com/LiveAdityaSingh/ML-Projects',
  },
  {
    name: 'Deep Learning Projects',
    desc: 'Deep learning experiments covering Computer Vision, NLP and neural architecture exploration with PyTorch and TensorFlow.',
    tags: ['PyTorch', 'TensorFlow', 'Computer Vision', 'NLP'],
    link: 'https://github.com/LiveAdityaSingh/Deep-Learning-Projects',
  },
  {
    name: 'UKFin Smart Renewal System',
    desc: 'A financial renewal automation system for UK markets, using predictive analytics and ML-driven decision logic for financial services.',
    tags: ['Python', 'ML', 'Financial AI', 'Automation'],
    link: 'https://github.com/LiveAdityaSingh/UKFin-Smart-Renewal-System',
  },
  {
    name: 'Data Analysis Results',
    desc: 'Exploratory data analysis and statistical modelling projects, including the winning FrontierTechAI Loyalty Program Challenge solution: a marginalized token-based reward ecosystem.',
    tags: ['Python', 'Pandas', 'NumPy', 'Visualization'],
    link: 'https://github.com/LiveAdityaSingh/Data-Anlaysis-Results',
  },
];

const CERTS = [
  { name: 'Google Cloud Professional Data Engineer', org: 'Google Cloud', year: 'Oct 2024' },
  { name: 'Artificial Intelligence Certification', org: 'ERACT Academy, IIT Kanpur', year: 'Jul 2022' },
  { name: 'Microsoft Certified: Azure AI Engineer Associate', org: 'Microsoft', year: 'Dec 2021' },
  { name: 'Neural Networks and Deep Learning', org: 'DeepLearning.AI', year: 'Aug 2021' },
  { name: 'Computer Vision Basics', org: 'University at Buffalo', year: 'Jan 2021' },
];

const BEYOND_WORK = [
  { title: 'HealthTech AI 2026', award: 'Winner', desc: 'Won the HealthTech AI 2026 competition with an optimized solution for doppler signal processing and denoising using a CNN, frame slicing, a sliding window algorithm and PyQTG. The project is live on GitHub.' },
  { title: 'FrontierTechAI Loyalty Program Challenge', award: 'Winner', desc: 'Built a marginalized token-based reward ecosystem and mobile app using Python, JSON, stable coin integration and banking APIs.' },
  { title: 'Dog NGO volunteer', desc: 'Co-ordinated volunteers with a dog NGO to rescue, feed and vaccinate dogs in the assigned region, which took teamwork, compassion and leadership.' },
  { title: 'Drummer and music enthusiast', desc: 'Passionate about rhythm and precision. Learning the drums and exploring where creative arts meet technology, with the same discipline I bring to engineering.' },
];

const ROLES = [
  { title: 'MLOps Engineer', desc: 'Deploying and monitoring ML models at production scale' },
  { title: 'AI / ML Engineer', desc: 'Building and shipping production AI/ML systems' },
  { title: 'Data Scientist', desc: 'Turning raw data into actionable insights' },
  { title: 'Data Engineer', desc: 'Scalable pipelines and cloud data infrastructure' },
];

/* ═══════════════════════════════════════════════════════════
   COMPONENTS
═══════════════════════════════════════════════════════════ */

function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('');

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 20);
      // Track every section so a nav link is only active inside its own section
      let current = '';
      for (let id of [...SECTION_IDS].reverse()) {
        const el = document.getElementById(id);
        if (el && window.scrollY >= el.offsetTop - 120) {
          current = id;
          break;
        }
      }
      setActiveSection(current);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <>
      <nav className={`nav${scrolled ? ' scrolled' : ''}`} aria-label="Main navigation">
        <a href="#home" className="nav-brand" aria-label="Aditya Narayan Singh, back to top">ANS</a>
        <ul className="nav-links">
          {NAV_LINKS.map(l => (
            <li key={l.href}>
              <a
                href={l.href}
                className={activeSection === l.href.slice(1) ? 'active' : ''}
                aria-current={activeSection === l.href.slice(1) ? 'true' : undefined}
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <button
          className="nav-mobile-btn"
          onClick={() => setDrawerOpen(o => !o)}
          aria-label={drawerOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={drawerOpen}
          aria-controls="nav-drawer"
        >
          {drawerOpen ? <X size={20} strokeWidth={1.5} /> : <Menu size={20} strokeWidth={1.5} />}
        </button>
      </nav>
      <div id="nav-drawer" className={`nav-drawer${drawerOpen ? ' open' : ''}`}>
        {NAV_LINKS.map(l => (
          <a key={l.href} href={l.href} onClick={() => setDrawerOpen(false)}>{l.label}</a>
        ))}
      </div>
    </>
  );
}

/* ─── Hero ─────────────────────────────────────────────── */
function Hero() {
  // One orchestrated entrance: the name first, then everything else together
  return (
    <div className="hero" id="home">
      <motion.div className="hero-grid" initial="hidden" animate="visible">
        <motion.div className="hero-photo" variants={fadeUp} custom={0.12}>
          <img
            src={process.env.PUBLIC_URL + '/avatar_transparent.png'}
            alt="Aditya Narayan Singh"
            width="500"
            height="440"
          />
        </motion.div>

        <div className="hero-content">
          <motion.div className="hero-badge" variants={fadeUp} custom={0.12}>
            Open to AI Engineer roles, UK
          </motion.div>

          <motion.h1 className="hero-name" variants={fadeUp} custom={0}>
            Aditya<br />Narayan Singh
          </motion.h1>

          <motion.div variants={fadeUp} custom={0.12}>
            <p className="hero-tagline">
              <span>AI Engineer</span> and <span>Data &amp; ML Engineer</span>, GCP &amp; Azure AI certified
            </p>

            <p className="hero-desc">
              3 years building production-grade AI and data solutions. Expert in architecting scalable
              pipelines and agentic AI systems using Python and Google Cloud (Vertex AI, BigQuery, Dataproc).
              Currently pursuing <strong>MSc AI &amp; Machine Learning</strong> at the University of Birmingham.
            </p>

            <div className="hero-ctas">
              <a href="#projects" className="btn-primary" id="hero-view-work-btn">
                View projects <ArrowRight size={16} strokeWidth={1.5} aria-hidden="true" />
              </a>
              <a href="#contact" className="btn-outline" id="hero-contact-btn">Get in touch</a>
              <a href={CV_URL} className="btn-outline btn-cv" id="hero-cv-link" download>
                <FileDown size={16} strokeWidth={1.5} aria-hidden="true" /> Download CV
              </a>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}

/* ─── About ────────────────────────────────────────────── */
function About() {
  return (
    <Section id="about" title="About">
      <div className="about-body">
        <p>
          Hi, I'm <strong>Aditya Narayan Singh</strong>, a GCP Professional Data Engineer and Azure AI Associate with 3 years of experience building production-grade AI and data solutions at Accenture. I'm based in <strong>Birmingham, UK</strong>.
        </p>
        <p>
          I have extensive experience designing scalable pipelines and agentic AI systems in Python and Google Cloud (Vertex AI, BigQuery, Dataproc). I cut system latency by 40% and introduced enterprise-level RAG and multi-agent frameworks.
        </p>
        <p>
          I'm studying <strong>MSc Artificial Intelligence &amp; Machine Learning</strong> at the University of Birmingham, with research interests in computer vision, deep learning and evolutionary computation.
        </p>
      </div>
      <dl className="about-facts">
        {ABOUT_FACTS.map(f => (
          <div className="about-fact" key={f.label}>
            <dt>{f.label}</dt>
            <dd>{f.value}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

function TimelineItem({ role, place, period, current, children }) {
  return (
    <div className={`timeline-item${current ? ' current' : ''}`}>
      <div className="timeline-dot" />
      <div className="timeline-header">
        <div>
          <h3 className="timeline-role">{role}</h3>
          <div className="timeline-company">{place}</div>
        </div>
        <span className="timeline-period">{period}</span>
      </div>
      <div className="timeline-body">{children}</div>
    </div>
  );
}

function Experience() {
  return (
    <Section id="experience" title="Experience">
      <div className="timeline">
        <TimelineItem role="Data Engineering Analyst" place="Accenture Solutions Pvt. Ltd., Pune, India" period="Oct 2022 – Aug 2025">
          <ul>
            <li><strong>40% lower architecture latency:</strong> architected and deployed end-to-end predictive pipelines using NiFi and GCP, accelerating data-driven decision-making.</li>
            <li><strong>300M+ records:</strong> engineered and optimised the ETL architecture, cutting data-to-model latency by 30% to enable real-time predictive analytics.</li>
            <li>Led technical discovery and requirement gathering for strategic code improvements, so AI solutions met rigorous business and infrastructure requirements.</li>
            <li>Presented complex ML enhancements and algorithm optimisations to business stakeholders as measurable ROI.</li>
          </ul>
        </TimelineItem>

        <TimelineItem role="Project Intern" place="Robust Results Pvt. Ltd., Kanpur, India" period="May 2020 – Jun 2020">
          <ul>
            <li><strong>90% forecasting accuracy:</strong> developed predictive analytics solutions using Python, Pandas and NumPy to support strategic risk assessment.</li>
            <li>Optimized an ensemble of classification and regression models to ensure production readiness across diverse enterprise datasets.</li>
          </ul>
        </TimelineItem>
      </div>
    </Section>
  );
}

function Projects() {
  return (
    <Section id="projects" title="Projects">
      <Reveal className="card-grid">
        {PROJECTS.map(p => (
          <article className={`project-card${p.featured ? ' featured' : ''}`} key={p.name}>
            {p.award && <span className="award-badge">{p.award}</span>}
            <h3 className="project-title">{p.name}</h3>
            <p className="project-desc">{p.desc}</p>
            <div className="chips">
              {p.tags.map(t => <span className="chip" key={t}>{t}</span>)}
            </div>
            <a href={p.link} className="project-link" target="_blank" rel="noopener noreferrer">
              View on GitHub <ArrowUpRight size={16} strokeWidth={1.5} aria-hidden="true" />
            </a>
          </article>
        ))}
      </Reveal>
    </Section>
  );
}

/* ─── Skills ─────────────────────────────────────────────── */
function Skills() {
  return (
    <Section id="skills" title="Skills">
      <div className="skills-list">
        {Object.entries(SKILLS).map(([group, items]) => (
          <div className="skill-group" key={group}>
            <h3 className="skill-group-title">{group}</h3>
            <div className="chips">
              {items.map(s => <span className="chip" key={s}>{s}</span>)}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Education() {
  return (
    <Section id="education" title="Education and certifications">
      <div className="timeline">
        <TimelineItem role="MSc Artificial Intelligence &amp; Machine Learning" place="University of Birmingham, Edgbaston, UK" period="Sep 2025 – Sep 2026" current>
          <p>Postgraduate programme in AI/ML with a focus on deep learning, computer vision, and evolutionary computation.</p>
          <ul>
            <li>Key Modules: Mathematics for AI, Artificial Intelligence, Computer Vision &amp; Images, NLP.</li>
            <li>Research interest: Multi-modal multi-agent architecture for clinical report generation with credibility score for trustworthy AI.</li>
          </ul>
        </TimelineItem>

        <TimelineItem role="B.Tech in Computer Science &amp; Engineering" place="Dr. APJ Abdul Kalam Technical University, India" period="Aug 2018 – Jul 2022">
          <p>Foundation in software engineering, algorithms, databases, and data systems.</p>
          <ul>
            <li>CGPA: <strong>7.58</strong></li>
            <li>Key Modules: Data Structures, Algorithm Design, Database Management Systems, AI/ML, Data Analytics.</li>
            <li>Final Year Project: Self Workout Trainer, a computer vision posture analysis system (published paper).</li>
          </ul>
        </TimelineItem>
      </div>

      <h3 className="subsection-title" id="certifications">Certifications</h3>
      <ul className="cert-list">
        {CERTS.map(c => (
          <li className="cert-item" key={c.name}>
            <span className="cert-name">{c.name}</span>
            <span className="cert-org">{c.org}, {c.year}</span>
          </li>
        ))}
      </ul>
      <p className="cert-note">
        Also completed professional certifications at Accenture in Tableau, Informatica, Power BI, Dataproc, PySpark and Communication.
      </p>
    </Section>
  );
}

function Publications() {
  return (
    <Section id="publications" title="Research">
      <div className="pub-card">
        <h3 className="pub-title">Self Workout Trainer: Skeletal Posture Analysis for Exercise Form Correction</h3>
        <p className="pub-journal">Academic publication, May 2022</p>
        <p className="pub-desc">
          Computer vision academic project implementing skeletal posture analysis to train a model
          that detects incorrect exercise form in real-time, raising alerts when the user's posture
          deviates from the correct form during workouts.
        </p>
      </div>
    </Section>
  );
}

function BeyondWork() {
  return (
    <Section id="cocurricular" title="Beyond work">
      <div className="beyond-list">
        {BEYOND_WORK.map(item => (
          <div className="beyond-item" key={item.title}>
            <h3 className="beyond-title">
              {item.title}
              {item.award && <span className="award-badge">{item.award}</span>}
            </h3>
            <p className="beyond-desc">{item.desc}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

/* ─── Contact ────────────────────────────────────────────── */
function Contact() {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [sent, setSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    const subject = encodeURIComponent(`Portfolio Contact from ${formData.name}`);
    const body = encodeURIComponent(`Name: ${formData.name}\nEmail: ${formData.email}\n\n${formData.message}`);
    window.location.href = `mailto:adityasingh2897@gmail.com?subject=${subject}&body=${body}`;
    setSent(true);
    setTimeout(() => setSent(false), 4000);
  };

  return (
    <Section id="contact" title="Contact">
      <p className="section-intro">
        Open to graduate roles, research collaborations, and interesting conversations.
        I'm actively seeking opportunities as I complete my MSc.
      </p>

      <div className="roles" id="roles">
        <h3 className="subsection-title">Roles I'm looking for</h3>
        <ul className="roles-list">
          {ROLES.map(r => (
            <li key={r.title}>
              <span className="role-title">{r.title}</span>
              <span className="role-desc">{r.desc}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="contact-layout">
        {/* Contact Form */}
        <form className="contact-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="form-input"
            placeholder="Your name"
            aria-label="Your name"
            autoComplete="name"
            value={formData.name}
            onChange={e => setFormData({ ...formData, name: e.target.value })}
            required
          />
          <input
            type="email"
            className="form-input"
            placeholder="Your email"
            aria-label="Your email"
            autoComplete="email"
            value={formData.email}
            onChange={e => setFormData({ ...formData, email: e.target.value })}
            required
          />
          <textarea
            className="form-textarea"
            placeholder="Your message"
            aria-label="Your message"
            value={formData.message}
            onChange={e => setFormData({ ...formData, message: e.target.value })}
            required
          />
          <button type="submit" className="form-submit" disabled={sent}>
            {sent ? 'Opening your email app…' : 'Send message'}
          </button>
        </form>

        {/* Contact Links */}
        <div className="contact-links">
          <a href="https://github.com/liveadityasingh" target="_blank" rel="noopener noreferrer" className="contact-card" id="contact-github">
            <Github size={20} strokeWidth={1.5} aria-hidden="true" />
            <div>
              <div className="contact-label">GitHub</div>
              <div className="contact-value">LiveAdityaSingh</div>
            </div>
          </a>

          <a href="https://www.linkedin.com/in/aditya-singhofficial/" target="_blank" rel="noopener noreferrer" className="contact-card" id="contact-linkedin">
            <Linkedin size={20} strokeWidth={1.5} aria-hidden="true" />
            <div>
              <div className="contact-label">LinkedIn</div>
              <div className="contact-value">aditya-singhofficial</div>
            </div>
          </a>

          <a href="mailto:adityasingh2897@gmail.com" className="contact-card" id="contact-email">
            <Mail size={20} strokeWidth={1.5} aria-hidden="true" />
            <div>
              <div className="contact-label">Email</div>
              <div className="contact-value">adityasingh2897@gmail.com</div>
            </div>
          </a>

          <a href={CV_URL} className="contact-card" id="contact-cv" download>
            <FileDown size={20} strokeWidth={1.5} aria-hidden="true" />
            <div>
              <div className="contact-label">CV</div>
              <div className="contact-value">Download PDF</div>
            </div>
          </a>

          <a href="tel:+447352674282" className="contact-card" id="contact-phone">
            <Phone size={20} strokeWidth={1.5} aria-hidden="true" />
            <div>
              <div className="contact-label">Phone</div>
              <div className="contact-value">+44 7352 674282</div>
            </div>
          </a>
        </div>
      </div>
    </Section>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>© 2026 Aditya Narayan Singh. Built with React.{' '}
        <a href="https://github.com/liveadityasingh" target="_blank" rel="noopener noreferrer">
          @LiveAdityaSingh
        </a>
      </p>
    </footer>
  );
}

/* ═══════════════════════════════════════════════════════════
   APP ROOT
═══════════════════════════════════════════════════════════ */
export default function App() {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const [showSphere, setShowSphere] = useState(false);

  // Mount the 3D background once the browser is idle
  useEffect(() => {
    if ('requestIdleCallback' in window) {
      const id = window.requestIdleCallback(() => setShowSphere(true), { timeout: 2000 });
      return () => window.cancelIdleCallback(id);
    }
    const timer = setTimeout(() => setShowSphere(true), 300);
    return () => clearTimeout(timer);
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <a href="#main" className="skip-link">Skip to content</a>
      {showSphere && (
        <Suspense fallback={null}>
          <AtomicBackground isChatbotOpen={isChatbotOpen} />
        </Suspense>
      )}
      <div className={`site${isChatbotOpen ? ' site--hidden' : ''}`} inert={isChatbotOpen}>
        <Navbar />
        <main className="page" id="main" tabIndex={-1}>
          <Hero />
          <About />
          <Experience />
          <Projects />
          <Skills />
          <Education />
          <Publications />
          <BeyondWork />
          <Contact />
        </main>
        <Footer />
      </div>
      <Chatbot isOpen={isChatbotOpen} setIsOpen={setIsChatbotOpen} />
    </MotionConfig>
  );
}
