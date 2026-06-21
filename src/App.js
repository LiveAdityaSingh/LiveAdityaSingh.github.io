import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Sphere, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import './index.css';
import Chatbot from './Chatbot';

/* ═══════════════════════════════════════════════════════════
   FRAMER MOTION VARIANTS
═══════════════════════════════════════════════════════════ */
const fadeUp = {
  hidden: { opacity: 0, y: 35 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.65, ease: 'easeOut' } },
};
const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const cardHover = {
  y: -4,
  transition: { duration: 0.3, ease: 'easeOut' },
};

/* ═══════════════════════════════════════════════════════════
   ANIMATED SECTION WRAPPER
═══════════════════════════════════════════════════════════ */
function AnimSection({ id, children, className }) {
  return (
    <motion.section
      id={id}
      className={`section ${className || ''}`}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
    >
      {children}
    </motion.section>
  );
}

function SectionHeader({ label, title }) {
  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <p className="section-label">{label}</p>
      <h2 className="section-title">{title}</h2>
      <div className="section-divider" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   DATA — All content preserved exactly
═══════════════════════════════════════════════════════════ */

const NAV_LINKS = [
  { href: '#about',         label: 'About'       },
  { href: '#experience',    label: 'Experience'  },
  { href: '#projects',      label: 'Projects'    },
  { href: '#skills',        label: 'Skills'      },
  { href: '#education',     label: 'Education'   },
  { href: '#publications',  label: 'Research'    },
  { href: '#certifications',label: 'Certs'       },
  { href: '#roles',         label: 'Roles'       },
  { href: '#contact',       label: 'Contact'     },
];

const SKILLS = {
  'AI / GenAI':    ['Multi-Agent Systems (ADK)', 'RAG', 'LLMs', 'NLP', 'Computer Vision', 'Prompt Engineering'],
  'ML / DL':       ['TensorFlow', 'PyTorch', 'Scikit-Learn', 'Keras', 'XGBoost', 'OpenCV'],
  'Data Eng.':     ['PySpark', 'Apache NiFi', 'dbt', 'Apache Kafka', 'Hadoop', 'Airflow', 'Informatica'],
  'Cloud':         ['GCP (Vertex AI, BigQuery, Dataproc)', 'Azure', 'AWS', 'Docker', 'Kubernetes', 'GitLab CI'],
  'Languages':     ['Python', 'SQL', 'JavaScript', 'R', 'Bash'],
  'Viz & BI':      ['Tableau', 'Power BI', 'Matplotlib', 'Seaborn'],
};

const PROJECTS = [
  {
    icon: '📡',
    name: 'Pulse-Doppler Signal Processing',
    desc: 'HealthTech AI 2026 🏆 winner. Optimized solution for doppler signal processing and denoising using CNN, frame slicing, sliding window algorithm and PyQTG.',
    tags: ['Python', 'CNN', 'PyQTG', 'Signal Processing'],
    link: 'https://github.com/LiveAdityaSingh/Pulse-Doppler-Signal-Processing',
    featured: true,
  },
  {
    icon: '🧠',
    name: 'MindGrow App',
    desc: 'An AI-powered productivity and growth application built with JavaScript, designed to help users track and develop cognitive habits.',
    tags: ['JavaScript', 'React', 'AI', 'Node.js'],
    link: 'https://github.com/LiveAdityaSingh/MindGrow-app',
  },
  {
    icon: '🤖',
    name: 'ML Projects Collection',
    desc: 'Comprehensive collection of machine learning projects including Mask Detection, Pose Estimation for workouts, Sentiment Analysis, and Toxic Comments Recognition models.',
    tags: ['Python', 'Scikit-Learn', 'TensorFlow', 'Jupyter'],
    link: 'https://github.com/LiveAdityaSingh/ML-Projects',
  },
  {
    icon: '🔬',
    name: 'Deep Learning Projects',
    desc: 'Portfolio of deep learning experiments covering Computer Vision, NLP, and Neural Architecture exploration with PyTorch and TensorFlow.',
    tags: ['PyTorch', 'TensorFlow', 'Computer Vision', 'NLP'],
    link: 'https://github.com/LiveAdityaSingh/Deep-Learning-Projects',
  },
  {
    icon: '💰',
    name: 'UKFin Smart Renewal System',
    desc: 'Financial renewal automation system for UK markets, implementing predictive analytics and ML-driven decision logic for financial services.',
    tags: ['Python', 'ML', 'Financial AI', 'Automation'],
    link: 'https://github.com/LiveAdityaSingh/UKFin-Smart-Renewal-System',
  },
  {
    icon: '📊',
    name: 'Data Analysis Results',
    desc: 'Exploratory data analysis and statistical modelling projects, including FrontierTechAI Loyalty Program Challenge winning solution — a marginalized token-based reward ecosystem.',
    tags: ['Python', 'Pandas', 'NumPy', 'Visualization'],
    link: 'https://github.com/LiveAdityaSingh/Data-Anlaysis-Results',
  },
];

const CERTS = [
  { icon: '☁️',  name: 'Microsoft Certified: Azure AI Engineer Associate',  org: 'Microsoft',          year: 'Dec 2021' },
  { icon: '🔷',  name: 'Google Cloud Professional Data Engineer',           org: 'Google Cloud',       year: 'Oct 2024' },
  { icon: '🎓',  name: 'Artificial Intelligence Certification',              org: 'ERACT Academy, IIT Kanpur', year: 'Jul 2022' },
  { icon: '🤖',  name: 'Neural Networks and Deep Learning',                  org: 'DeepLearning.AI',    year: 'Aug 2021' },
  { icon: '👁️',  name: 'Computer Vision Basics',                             org: 'University at Buffalo', year: 'Jan 2021' },
];

const ROLES = [
  { icon: '⚙️', title: 'MLOps Engineer',        desc: 'Deploying & monitoring ML models at production scale', rank: '1st' },
  { icon: '🤖', title: 'AI / ML Engineer',       desc: 'Building and shipping production AI/ML systems',       rank: '2nd' },
  { icon: '📊', title: 'Data Scientist',          desc: 'Turning raw data into actionable insights',            rank: '3rd' },
  { icon: '📡', title: 'Data Engineer',           desc: 'Scalable pipelines & cloud data infrastructure',       rank: '4th' },
];

/* ═══════════════════════════════════════════════════════════
   PINNED 3D GLOBE + STARFIELD CANVAS
   - Globe auto-rotates slowly
   - Camera dollies back as user scrolls (scroll-linked)
   - Starfield + atmosphere glow rendered on same canvas
═══════════════════════════════════════════════════════════ */
function Earth() {
  const scrollRef = useRef(0);
  const earthRef = useRef();
  const cloudsRef = useRef();

  const [colorMap, emissiveMap, bumpMap, cloudsMap] = useTexture([
    'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
    'https://unpkg.com/three-globe/example/img/earth-night.jpg',
    'https://unpkg.com/three-globe/example/img/earth-topology.png',
    'https://unpkg.com/three-globe/example/img/earth-clouds1024.png'
  ]);

  useEffect(() => {
    const onScroll = () => {
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      scrollRef.current = maxScroll > 0 ? window.scrollY / maxScroll : 0;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.05;
    }
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y += delta * 0.07;
    }
    // Scroll-linked dolly
    const CAM_NEAR = 2.5;
    const CAM_FAR = 6.0;
    const targetZ = CAM_NEAR + (CAM_FAR - CAM_NEAR) * scrollRef.current;
    state.camera.position.z += (targetZ - state.camera.position.z) * 0.05;
  });

  return (
    <group position={[0, -0.2, 0]}>
      <ambientLight intensity={0.1} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      
      {/* Earth */}
      <Sphere ref={earthRef} args={[1, 64, 64]}>
        <meshStandardMaterial 
          map={colorMap}
          emissiveMap={emissiveMap}
          emissive={new THREE.Color(0xffffff)}
          emissiveIntensity={0.8}
          bumpMap={bumpMap}
          bumpScale={0.02}
        />
      </Sphere>

      {/* Clouds */}
      <Sphere ref={cloudsRef} args={[1.006, 64, 64]}>
        <meshStandardMaterial 
          map={cloudsMap}
          transparent={true}
          opacity={0.4}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </Sphere>

      {/* Atmosphere Glow */}
      <Sphere args={[1.04, 64, 64]}>
        <meshBasicMaterial 
          color="#3ecb7a" 
          transparent={true} 
          opacity={0.15} 
          blending={THREE.AdditiveBlending}
          side={THREE.BackSide}
        />
      </Sphere>
    </group>
  );
}

function GlobeBackground() {
  return (
    <div className="starfield-canvas" style={{ zIndex: 0, position: 'fixed', inset: 0, pointerEvents: 'none' }}>
      <Canvas camera={{ position: [0, 0, 2.5], fov: 45 }}>
        <color attach="background" args={['#060507']} />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <React.Suspense fallback={null}>
          <Earth />
        </React.Suspense>
      </Canvas>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   3D GLASS SHARD BACKGROUND + DATA STREAMS
═══════════════════════════════════════════════════════════ */
const SHARDS = [
  { type: 'hex',  w: 140, h: 140, top: '8%',  left: '6%',   rx: 15,  ry: -20, rz: 5,   dur: 22, del: 0,  op: 0.4 },
  { type: 'hex',  w: 80,  h: 80,  top: '22%', right: '12%', rx: -12, ry: 18,  rz: -6,  dur: 28, del: 3,  op: 0.3 },
  { type: 'hex',  w: 55,  h: 55,  top: '55%', left: '3%',   rx: 8,   ry: -10, rz: 12,  dur: 25, del: 5,  op: 0.25 },
  { type: 'hex',  w: 100, h: 100, top: '75%', right: '8%',  rx: -18, ry: 12,  rz: -4,  dur: 30, del: 8,  op: 0.35 },
  { type: 'hex',  w: 40,  h: 40,  top: '42%', left: '52%',  rx: 20,  ry: -5,  rz: 8,   dur: 18, del: 2,  op: 0.2 },
  { type: 'pane', w: 170, h: 110, top: '14%', right: '4%',  rx: -8,  ry: 25,  rz: -3,  dur: 26, del: 4,  op: 0.25 },
  { type: 'pane', w: 110, h: 70,  top: '65%', left: '13%',  rx: 12,  ry: -22, rz: 6,   dur: 24, del: 7,  op: 0.3  },
  { type: 'pane', w: 90,  h: 60,  top: '35%', right: '26%', rx: -6,  ry: 14,  rz: -10, dur: 20, del: 1,  op: 0.2  },
  { type: 'pane', w: 60,  h: 40,  top: '88%', left: '42%',  rx: 15,  ry: -8,  rz: 3,   dur: 22, del: 6,  op: 0.18 },
  { type: 'hex',  w: 70,  h: 70,  top: '3%',  left: '38%',  rx: -10, ry: 20,  rz: -7,  dur: 32, del: 10, op: 0.22 },
];

function CyberBackground() {
  return (
    <div className="cyber-bg">
      <div className="ambient-orb ambient-orb-1" />
      <div className="ambient-orb ambient-orb-2" />
      <div className="ambient-orb ambient-orb-3" />
      {SHARDS.map((s, i) => (
        <div
          key={i}
          className={`glass-shard ${s.type === 'hex' ? 'glass-shard-hex' : 'glass-shard-pane'}`}
          style={{
            width: s.w, height: s.h,
            top: s.top, left: s.left, right: s.right, bottom: s.bottom,
            '--rx': `${s.rx}deg`, '--ry': `${s.ry}deg`, '--rz': `${s.rz}deg`,
            '--duration': `${s.dur}s`, '--delay': `${s.del}s`, '--op': s.op,
          }}
        >
          <div className="glass-shard-inner" />
        </div>
      ))}
      {/* Data streams */}
      <div className="data-stream data-stream-1"><div className="data-stream-line" /></div>
      <div className="data-stream data-stream-2"><div className="data-stream-line" /></div>
      <div className="data-stream data-stream-3"><div className="data-stream-line" /></div>
      <div className="data-stream data-stream-4"><div className="data-stream-vert" style={{ height: '100%' }} /></div>
      <div className="data-stream data-stream-5"><div className="data-stream-vert" style={{ height: '100%' }} /></div>
    </div>
  );
}

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
      const sections = NAV_LINKS.map(l => l.href.slice(1));
      for (let id of [...sections].reverse()) {
        const el = document.getElementById(id);
        if (el && window.scrollY >= el.offsetTop - 120) {
          setActiveSection(id);
          break;
        }
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <>
      <nav className={`nav${scrolled ? ' scrolled' : ''}`} role="navigation" aria-label="Main navigation">
        <div className="nav-brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>ANS</div>
        <ul className="nav-links">
          {NAV_LINKS.map(l => (
            <li key={l.href}>
              <a href={l.href} className={activeSection === l.href.slice(1) ? 'active' : ''} onClick={() => setDrawerOpen(false)}>
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <button className="nav-mobile-btn" onClick={() => setDrawerOpen(o => !o)} aria-label="Open menu">
          {drawerOpen ? '✕' : '☰'}
        </button>
      </nav>
      <div className={`nav-drawer${drawerOpen ? ' open' : ''}`}>
        {NAV_LINKS.map(l => (
          <a key={l.href} href={l.href} onClick={() => setDrawerOpen(false)}>{l.label}</a>
        ))}
      </div>
    </>
  );
}

/* ─── Hero with Holographic Bio Card ─────────────────────── */
function Hero() {
  return (
    <div className="hero" id="home">
      <div className="hero-bg">
        <div className="hero-blob hero-blob-1" />
        <div className="hero-blob hero-blob-2" />
        <div className="hero-blob hero-blob-3" />
      </div>
      <div className="hero-grid">
        {/* Holographic photo */}
        <motion.div
          className="hero-photo-container"
          initial={{ opacity: 0, scale: 0.8, rotateY: -15 }}
          animate={{ opacity: 1, scale: 1, rotateY: 0 }}
          transition={{ duration: 0.9, delay: 0.2, ease: 'easeOut' }}
        >
          <div className="hero-photo-hex">
            <img
              src="https://avatars.githubusercontent.com/liveadityasingh"
              alt="Aditya Narayan Singh"
            />
          </div>
        </motion.div>

        {/* Bio content */}
        <div className="hero-content">
          <motion.div
            className="hero-badge"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <span className="hero-badge-dot" />
            Open to opportunities · 2026
          </motion.div>

          <motion.h1
            className="hero-name"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.45 }}
          >
            Aditya<br />Narayan Singh
          </motion.h1>

          <motion.p
            className="hero-tagline"
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <span>GCP &amp; Azure AI</span> certified <span>Data &amp; ML Engineer</span>
          </motion.p>

          <motion.p
            className="hero-desc"
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.75 }}
          >
            2+ years building production-grade AI and data solutions. Expert in architecting scalable
            pipelines and agentic AI systems using Python and Google Cloud (Vertex AI, BigQuery, Dataproc).
            Currently pursuing <strong>MSc AI &amp; Machine Learning</strong> at the University of Birmingham.
          </motion.p>

          <motion.div
            className="hero-ctas"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.9 }}
          >
            <a href="#projects" className="btn-primary" id="hero-view-work-btn">View My Work →</a>
            <a href="#contact" className="btn-outline" id="hero-contact-btn">Get In Touch</a>
          </motion.div>
        </div>

        {/* Stats row */}
        <motion.div
          className="hero-stats"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.05 }}
        >
          <div>
            <span className="hero-stat-num">2+</span>
            <span className="hero-stat-label">Years Experience</span>
          </div>
          <div>
            <span className="hero-stat-num">6</span>
            <span className="hero-stat-label">GitHub Projects</span>
          </div>
          <div>
            <span className="hero-stat-num">2×</span>
            <span className="hero-stat-label">Award Winner</span>
          </div>
          <div>
            <span className="hero-stat-num">5</span>
            <span className="hero-stat-label">Certifications</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

/* ─── About (text-only, photo is in Hero now) ────────────── */
function About() {
  return (
    <AnimSection id="about">
      <SectionHeader label="// 01 · About" title="Who I Am" />
      <motion.div
        className="about-panel"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <p>
          Hi, I'm <strong>Aditya Narayan Singh</strong> — a GCP Professional Data Engineer and
          Azure AI Associate with 2+ years of experience building production-grade AI and data
          solutions at Accenture. Based in <strong>Birmingham, UK</strong>.
        </p>
        <p>
          I am an expert in architecting scalable pipelines and agentic AI systems using Python
          and Google Cloud (Vertex AI, BigQuery, Dataproc). My proven track record includes
          reducing system latency by 40% and deploying enterprise-grade RAG and multi-agent frameworks.
        </p>
        <p>
          Currently pursuing my <strong>MSc in Artificial Intelligence &amp; Machine Learning</strong> at the
          University of Birmingham, with research interests at the intersection of evolutionary
          computation, deep learning, and computer vision.
        </p>
        <div className="about-highlights">
          {[
            '📍 Birmingham, UK',
            '🎓 MSc Artificial Intelligence & Machine Learning — University of Birmingham (2025–2026)',
            '💼 Ex-Data Engineering Analyst @ Accenture (Oct 2022 – Aug 2025)',
            '🏆 HealthTech AI 2026 Winner · FrontierTechAI Loyalty Program Winner',
            '🥁 Drummer, volunteer dog rescuer & multilingual (English, Hindi, Punjabi, German)',
          ].map(h => (
            <motion.div
              className="about-highlight"
              key={h}
              initial={{ opacity: 0, x: -15 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <div className="about-highlight-dot" />
              <span>{h}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </AnimSection>
  );
}

function Experience() {
  return (
    <AnimSection id="experience">
      <SectionHeader label="// 02 · Experience" title="Work History" />
      <div className="timeline">
        <motion.div className="timeline-item" variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}>
          <div className="timeline-dot" />
          <div className="timeline-header">
            <div>
              <div className="timeline-role">Data Engineering Analyst</div>
              <div className="timeline-company">Accenture Solutions Pvt. Ltd. · Pune, India</div>
            </div>
            <span className="timeline-period">Oct 2022 – Aug 2025</span>
          </div>
          <div className="timeline-body">
            <p>Architected and delivered end-to-end predictive data pipelines for high-value enterprise clients using GCP and Apache NiFi.</p>
            <ul>
              <li>Architected and deployed end-to-end predictive pipelines using <strong>NiFi and GCP</strong>, reducing architecture latency by <strong>40%</strong> and accelerating data-driven decision-making.</li>
              <li>Engineered and optimised ETL architecture for <strong>300M+ records</strong>, cutting data-to-model latency by <strong>30%</strong> to enable real-time predictive analytics.</li>
              <li>Led technical discovery and requirement gathering for strategic code improvements, ensuring AI solutions met rigorous business and infrastructure requirements.</li>
              <li>Delivered high-impact presentations translating complex ML enhancements and algorithm optimisations into measurable ROI for business stakeholders.</li>
              <li>Completed professional certifications in Tableau, Informatica, PowerBI, DataProc, PySpark, and Communication.</li>
            </ul>
          </div>
        </motion.div>

        <motion.div className="timeline-item" variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}>
          <div className="timeline-dot" />
          <div className="timeline-header">
            <div>
              <div className="timeline-role">Project Intern</div>
              <div className="timeline-company">Robust Results Pvt. Ltd. · Kanpur, India</div>
            </div>
            <span className="timeline-period">May 2020 – Jun 2020</span>
          </div>
          <div className="timeline-body">
            <p>Developed predictive analytics solutions for enterprise risk assessment.</p>
            <ul>
              <li>Built predictive analytics solutions using Python, Pandas, and NumPy, achieving <strong>90% forecasting accuracy</strong> for strategic risk assessment.</li>
              <li>Optimised ensembles of classification/regression models ensuring production-ready performance across diverse enterprise datasets.</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </AnimSection>
  );
}

function Projects() {
  return (
    <AnimSection id="projects">
      <SectionHeader label="// 03 · Projects" title="GitHub Projects" />
      <motion.div className="card-grid" variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
        {PROJECTS.map(p => (
          <motion.div className="project-card" key={p.name} variants={fadeUp} whileHover={cardHover}>
            <div className="project-header">
              <div className="project-icon" aria-hidden="true">{p.icon}</div>
              {p.featured && <span className="project-award-badge">🏆 AWARD</span>}
            </div>
            <div className="project-title">{p.name}</div>
            <p className="project-desc">{p.desc}</p>
            <div className="project-tags">
              {p.tags.map(t => <span className="project-tag" key={t}>{t}</span>)}
            </div>
            <a href={p.link} className="project-link" target="_blank" rel="noopener noreferrer">
              View on GitHub →
            </a>
          </motion.div>
        ))}
      </motion.div>
    </AnimSection>
  );
}

/* ─── Skills — Luminous Orb Cards ────────────────────────── */
function Skills() {
  const orbDelays = [0, 1, 2, 0.5, 1.5, 2.5];
  const orbDurations = [6, 7, 8, 6.5, 7.5, 5.5];

  return (
    <AnimSection id="skills">
      <SectionHeader label="// 04 · Skills" title="Technical Stack" />
      <motion.div
        className="skills-orbit"
        variants={stagger}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '-60px' }}
      >
        {Object.entries(SKILLS).map(([group, items], i) => (
          <motion.div
            className="skill-orb"
            key={group}
            variants={fadeUp}
            style={{
              '--float-dur': `${orbDurations[i]}s`,
              '--float-del': `${orbDelays[i]}s`,
            }}
            whileHover={{
              scale: 1.03,
              borderColor: 'rgba(0, 240, 255, 0.3)',
              transition: { duration: 0.3 },
            }}
          >
            <div className="skill-orb-title">{group}</div>
            <div className="skill-orb-tags">
              {items.map(s => <span className="skill-orb-tag" key={s}>{s}</span>)}
            </div>
          </motion.div>
        ))}
      </motion.div>
    </AnimSection>
  );
}

function Education() {
  return (
    <AnimSection id="education">
      <SectionHeader label="// 05 · Education" title="Academic Background" />
      <div className="timeline">
        <motion.div className="timeline-item" variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}>
          <div className="timeline-dot" />
          <div className="timeline-header">
            <div>
              <div className="timeline-role">MSc Artificial Intelligence &amp; Machine Learning</div>
              <div className="timeline-company">University of Birmingham, Edgbaston · UK</div>
            </div>
            <span className="timeline-period">Sep 2025 – Sep 2026</span>
          </div>
          <div className="timeline-body">
            <p>Postgraduate programme in AI/ML with a focus on deep learning, computer vision, and evolutionary computation.</p>
            <ul>
              <li>Key Modules: Mathematics for AI, Machine Learning, Artificial Intelligence, Computer Vision &amp; Imagery, NLP.</li>
              <li>Research interest: Optimization of Evolutionary Computation techniques for deep learning models.</li>
            </ul>
          </div>
        </motion.div>

        <motion.div className="timeline-item" variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }}>
          <div className="timeline-dot" />
          <div className="timeline-header">
            <div>
              <div className="timeline-role">B.Tech in Computer Science &amp; Engineering</div>
              <div className="timeline-company">Dr. APJ Abdul Kalam Technical University · India</div>
            </div>
            <span className="timeline-period">Aug 2018 – Jul 2022</span>
          </div>
          <div className="timeline-body">
            <p>Foundation in software engineering, algorithms, databases, and data systems.</p>
            <ul>
              <li>CGPA: <strong>7.29</strong></li>
              <li>Key Modules: Data Structures, Algorithm Design, Database Management Systems, AOM, Data Analytics.</li>
              <li>Final Year Project: Self Workout Trainer — computer vision posture analysis system (published paper).</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </AnimSection>
  );
}

function Publications() {
  return (
    <AnimSection id="publications">
      <SectionHeader label="// 06 · Research" title="Publications" />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <motion.div
          className="pub-card"
          whileHover={{ x: 4, boxShadow: '0 0 15px rgba(123,45,255,0.25), 0 0 45px rgba(123,45,255,0.08)' }}
          transition={{ duration: 0.3 }}
        >
          <p className="pub-title">"Self Workout Trainer — Skeletal Posture Analysis for Exercise Form Correction"</p>
          <p className="pub-journal">Academic Publication · May 2022</p>
          <p style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.65' }}>
            Computer vision academic project implementing skeletal posture analysis to train a model
            that detects incorrect exercise form in real-time, raising alerts when the user's posture
            deviates from the correct form during workouts.
          </p>
        </motion.div>
      </div>
    </AnimSection>
  );
}

function Certifications() {
  return (
    <AnimSection id="certifications">
      <SectionHeader label="// 07 · Certifications" title="Credentials" />
      <motion.div className="cert-grid" variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
        {CERTS.map(c => (
          <motion.div className="cert-card" key={c.name} variants={fadeUp} whileHover={cardHover}>
            <div className="cert-icon">{c.icon}</div>
            <div>
              <div className="cert-name">{c.name}</div>
              <div className="cert-org">{c.org} · {c.year}</div>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </AnimSection>
  );
}

function CoCurricular() {
  return (
    <AnimSection id="cocurricular">
      <SectionHeader label="// 08 · Beyond Code" title="Co-Curricular &amp; Achievements" />
      <motion.div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }} variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
        {[
          { icon: '🏆', title: 'HealthTech AI 2026 — Competition Winner', desc: 'Won the HealthTech AI 2026 competition by providing an optimized solution for doppler signal processing and denoising using CNN, frame slicing, sliding window algorithm and PyQTG. The project is live on GitHub.' },
          { icon: '🎯', title: 'FrontierTechAI Loyalty Program Challenge — Winner', desc: 'Built a marginalized token-based reward ecosystem and mobile app using Python, JSON, stable coin integration and banking APIs, winning the FrontierTechAI Loyalty Program Challenge.' },
          { icon: '🐾', title: 'Dog NGO Volunteer', desc: 'Co-ordinated volunteers while associated with a dog NGO to rescue, feed and vaccinate dogs in the assigned region — demonstrating teamwork, compassion and leadership skills.' },
          { icon: '🥁', title: 'Drummer & Music Enthusiast', desc: 'Passionate about rhythm and precision. Learning the drums and exploring the intersection of creative arts and technology — the same discipline I bring to every engineering challenge.' },
        ].map(item => (
          <motion.div className="cocurr-card" key={item.title} variants={fadeUp} whileHover={cardHover}>
            <div className="cocurr-icon">{item.icon}</div>
            <div>
              <div className="cocurr-title">{item.title}</div>
              <p className="cocurr-desc">{item.desc}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </AnimSection>
  );
}

function AspiredRoles() {
  return (
    <AnimSection id="roles">
      <SectionHeader label="// 09 · Ambitions" title="Aspired Roles" />
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '1.75rem', maxWidth: '560px' }}>
        Actively seeking opportunities as I complete my MSc.
      </p>
      <motion.div className="roles-grid" variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
        {ROLES.map(r => (
          <motion.div className="role-card" key={r.title} variants={fadeUp} whileHover={cardHover}>
            <div className="role-icon">{r.icon}</div>
            <div className="role-title">{r.title}</div>
            <div className="role-desc">{r.desc}</div>
          </motion.div>
        ))}
      </motion.div>
    </AnimSection>
  );
}

/* ─── Contact — Neon Wireframe Form + Links ──────────────── */
function Contact() {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });

  const handleSubmit = (e) => {
    e.preventDefault();
    const subject = encodeURIComponent(`Portfolio Contact from ${formData.name}`);
    const body = encodeURIComponent(`Name: ${formData.name}\nEmail: ${formData.email}\n\n${formData.message}`);
    window.location.href = `mailto:adityasingh2897@gmail.com?subject=${subject}&body=${body}`;
  };

  return (
    <AnimSection id="contact">
      <SectionHeader label="// 10 · Contact" title="Get In Touch" />
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', maxWidth: '500px', lineHeight: '1.7' }}>
        Open to graduate roles, research collaborations, and interesting conversations.
      </p>

      <div className="contact-layout">
        {/* Contact Form */}
        <motion.form
          className="contact-form"
          onSubmit={handleSubmit}
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="form-group">
            <input
              type="text"
              className="form-input"
              placeholder="Your Name"
              value={formData.name}
              onChange={e => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <input
              type="email"
              className="form-input"
              placeholder="Your Email"
              value={formData.email}
              onChange={e => setFormData({ ...formData, email: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <textarea
              className="form-textarea"
              placeholder="Your Message"
              value={formData.message}
              onChange={e => setFormData({ ...formData, message: e.target.value })}
              required
            />
          </div>
          <button type="submit" className="form-submit">Send Message →</button>
        </motion.form>

        {/* Contact Links */}
        <motion.div
          className="contact-links"
          variants={stagger}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.a href="https://github.com/liveadityasingh" target="_blank" rel="noopener noreferrer" className="contact-card" id="contact-github" variants={fadeUp} whileHover={cardHover}>
            <div className="contact-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z"/>
              </svg>
            </div>
            <div>
              <div className="contact-label">GitHub</div>
              <div className="contact-value">LiveAdityaSingh</div>
            </div>
          </motion.a>

          <motion.a href="https://www.linkedin.com/in/aditya-singhofficial/" target="_blank" rel="noopener noreferrer" className="contact-card" id="contact-linkedin" variants={fadeUp} whileHover={cardHover}>
            <div className="contact-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
              </svg>
            </div>
            <div>
              <div className="contact-label">LinkedIn</div>
              <div className="contact-value">aditya-singhofficial</div>
            </div>
          </motion.a>

          <motion.a href="mailto:adityasingh2897@gmail.com" className="contact-card" id="contact-email" variants={fadeUp} whileHover={cardHover}>
            <div className="contact-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                <polyline points="22,6 12,13 2,6"/>
              </svg>
            </div>
            <div>
              <div className="contact-label">Email</div>
              <div className="contact-value">adityasingh2897@gmail.com</div>
            </div>
          </motion.a>

          <motion.a href="tel:+447357674282" className="contact-card" id="contact-phone" variants={fadeUp} whileHover={cardHover}>
            <div className="contact-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 13a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.6 2.22h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L7.91 9.91a16 16 0 0 0 6.18 6.18l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
              </svg>
            </div>
            <div>
              <div className="contact-label">Phone</div>
              <div className="contact-value">+44 7357 674282</div>
            </div>
          </motion.a>
        </motion.div>
      </div>
    </AnimSection>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>© 2026 · Aditya Narayan Singh · Built with React · {' '}
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
  return (
    <>
      <GlobeBackground />
      <CyberBackground />
      <Navbar />
      <main className="page">
        <Hero />
        <About />
        <Experience />
        <Projects />
        <Skills />
        <Education />
        <Publications />
        <Certifications />
        <CoCurricular />
        <AspiredRoles />
        <Contact />
      </main>
      <Footer />
      <Chatbot />
    </>
  );
}
