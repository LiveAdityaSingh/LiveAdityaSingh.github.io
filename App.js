import React from 'react';
import { motion } from 'framer-motion';
import { Database, Code, Award, BookOpen, Briefcase, Cpu } from 'lucide-react';

const Section = ({ id, title, icon: Icon, children }) => (
  <motion.section
    id={id}
    initial={{ opacity: 0, y: 50 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.8, ease: "easeOut" }}
    viewport={{ once: false, amount: 0.5 }}
    className="min-h-screen flex flex-col justify-center px-8 md:px-24 py-20 border-l border-blue-500/30 ml-4 md:ml-20 relative"
  >
    <div className="absolute -left-[9px] top-1/2 w-4 h-4 bg-blue-500 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.8)]" />
    <div className="flex items-center gap-4 mb-8">
      <Icon className="text-blue-400" size={32} />
      <h2 className="text-4xl font-bold tracking-tight uppercase">{title}</h2>
    </div>
    {children}
  </motion.section>
);

export default function Portfolio() {
  return (
    <div className="relative">
      <div className="mesh-line" />
      
      {/* Hero Section */}
      <section className="h-screen flex flex-col justify-center items-center text-center px-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1 }}
        >
          <h1 className="text-6xl md:text-8xl font-extrabold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            AI & DATA ENGINEER
          </h1>
          <p className="mt-4 text-xl text-slate-400 font-mono tracking-widest">
            POSTGRADUATE PORTFOLIO // DATA PIPELINING // MACHINE LEARNING
          </p>
        </motion.div>
        <div className="mt-20 animate-bounce text-blue-400">↓ Scroll to traverse data points</div>
      </section>

      {/* Experience / Data Engineering */}
      <Section id="experience" title="Work Experience" icon={Briefcase}>
        <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800 backdrop-blur-md">
          <h3 className="text-2xl font-bold text-blue-300">Data Engineer @ Accenture</h3>
          <p className="text-slate-400 mb-4 italic">Collaborated in cross-functional teams to build scalable ETL pipelines.</p>
          <ul className="list-disc list-inside space-y-2 text-slate-300">
            <li>Optimized SQL queries reducing processing time by 30%.</li>
            <li>Architected data lakes for high-volume enterprise clients.</li>
          </ul>
        </div>
      </Section>

      {/* Projects */}
      <Section id="projects" title="AI & ML Projects" icon={Cpu}>
        <div className="grid md:grid-cols-2 gap-6">
          {[
            { name: "SME Lending AI", desc: "Predictive modeling for small business credit risk." },
            { name: "NLP Sentiment Engine", desc: "Real-time analysis of streaming financial news." }
          ].map((proj) => (
            <div key={proj.name} className="p-6 border border-slate-800 bg-slate-900/40 rounded-lg hover:border-blue-500/50 transition-colors">
              <h4 className="text-xl font-bold mb-2">{proj.name}</h4>
              <p className="text-slate-400">{proj.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Skills */}
      <Section id="skills" title="Technical Skills" icon={Code}>
        <div className="flex flex-wrap gap-4">
          {["Python", "PySpark", "AWS", "SQL", "TensorFlow", "React", "Hadoop", "Scikit-Learn"].map(skill => (
            <span key={skill} className="px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-300 font-mono">
              {skill}
            </span>
          ))}
        </div>
      </Section>

      {/* Publications */}
      <Section id="publications" title="Publications" icon={BookOpen}>
        <div className="space-y-6">
          <div className="border-l-2 border-emerald-500 pl-4">
            <h4 className="text-xl font-semibold italic">"Optimization of Evolutionary Computation in ML Models"</h4>
            <p className="text-slate-400">International Journal of AI Research, 2025</p>
          </div>
        </div>
      </Section>

      {/* Achievements & Certifications */}
      <Section id="achievements" title="Achievements" icon={Award}>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-blue-400 mb-4 font-bold">Awards</h3>
            <p className="text-slate-300 underline">Postgraduate Representative (EPS Department)</p>
          </div>
          <div>
            <h3 className="text-blue-400 mb-4 font-bold">Certifications</h3>
            <ul className="text-slate-300 space-y-1">
              <li>• AWS Certified Solutions Architect</li>
              <li>• Google Professional Data Engineer</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Co-Curricular */}
      <Section id="cocurricular" title="Co-Curricular" icon={Database}>
        <div className="p-6 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
          <p className="text-lg text-slate-300 leading-relaxed">
            Beyond the screen, I am passionate about rhythm and precision—currently 
            learning the **drums** and exploring the intersection of creative arts and technology.
          </p>
        </div>
      </Section>

      <footer className="py-10 text-center text-slate-600 text-sm">
        © 2026 | Built with React & Tailwind
      </footer>
    </div>
  );
}
