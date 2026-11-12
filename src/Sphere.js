import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useReducedMotion } from 'framer-motion';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { ShaderMaterial, Color, NormalBlending } from 'three';

/* ═══════════════════════════════════════════════════════════
   3D ATOMIC PARTICLE SPHERE
   - ~700 small particles distributed via fibonacci sphere
   - Auto-rotates slowly, tilts slightly opposite to the mouse
   - Fades out past the hero and stops rendering when idle
═══════════════════════════════════════════════════════════ */
const PARTICLE_COUNT = 700;
const CHAT_PARTICLE_COUNT = 280;
const SPHERE_RADIUS = 2.52;
const CAMERA_Z = 6.5;

function ParticleSphere({ mouseRef, isChatbotOpen, animate }) {
  const pointsRef = useRef();
  const tiltRef = useRef({ x: 0, y: 0 });
  const scaleRef = useRef(1.0);
  const invalidate = useThree((state) => state.invalidate);
  const camera = useThree((state) => state.camera);
  const aspect = useThree((state) => state.size.width / state.size.height);

  // On portrait screens pull the camera back so the sphere fits the width
  useEffect(() => {
    camera.position.z = CAMERA_Z / Math.min(1, aspect);
    invalidate();
  }, [camera, aspect, invalidate]);

  // Generate fibonacci-distributed points on a sphere and shuffle them
  const { positions, sizes } = useMemo(() => {
    const points = [];
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Fibonacci sphere distribution
      const theta = (2 * Math.PI * i) / goldenRatio;
      const phi = Math.acos(1 - (2 * (i + 0.5)) / PARTICLE_COUNT);

      // Add very subtle variation to radius for an organic but cleaner boundary
      const r = SPHERE_RADIUS * (0.985 + Math.random() * 0.03);

      points.push({
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi),
        size: 0.4 + Math.random() * 0.7
      });
    }

    // Shuffle points to ensure drawRange truncates randomly distributed particles
    for (let i = points.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [points[i], points[j]] = [points[j], points[i]];
    }

    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const sz = new Float32Array(PARTICLE_COUNT);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pos[i * 3] = points[i].x;
      pos[i * 3 + 1] = points[i].y;
      pos[i * 3 + 2] = points[i].z;
      sz[i] = points[i].size;
    }

    return { positions: pos, sizes: sz };
  }, []);

  // When the render loop is paused, draw one frame so state changes still show
  useEffect(() => {
    invalidate();
  }, [invalidate, isChatbotOpen, animate]);

  useFrame((state, delta) => {
    const points = pointsRef.current;
    if (!points) return;

    // While paused there is no next frame to ease into, so snap to the targets
    const ease = animate ? 0.05 : 1;

    if (animate) {
      // Auto-rotate slowly
      points.rotation.y += Math.min(delta, 0.1) * 0.03;

      // Smooth tilt toward target (opposite of mouse direction)
      const targetX = -mouseRef.current.y * 0.06;
      const targetY = -mouseRef.current.x * 0.06;
      tiltRef.current.x += (targetX - tiltRef.current.x) * 0.03;
      tiltRef.current.y += (targetY - tiltRef.current.y) * 0.03;
      points.rotation.x = tiltRef.current.x;
      points.rotation.z = tiltRef.current.y * 0.3;
    }

    // Move to the upper half and thin out while the chatbot is open
    const targetPosY = isChatbotOpen ? 1.2 : 0.0;
    points.position.y += (targetPosY - points.position.y) * ease;
    points.geometry.setDrawRange(0, isChatbotOpen ? CHAT_PARTICLE_COUNT : PARTICLE_COUNT);

    const targetScale = isChatbotOpen ? 0.5 : 1.0;
    scaleRef.current += (targetScale - scaleRef.current) * ease;

    // Gentle pulse via scale
    const pulse = animate ? Math.sin(state.clock.elapsedTime * 0.5) * 0.008 : 0;
    points.scale.setScalar(scaleRef.current * (1 + pulse));

    // Pass scale to shader so particles shrink proportionally
    points.material.uniforms.uScale.value = scaleRef.current;
  });

  // Custom shader material for soft round particles
  const shaderMaterial = useMemo(() => {
    return new ShaderMaterial({
      uniforms: {
        uScale: { value: 1.0 },
        uColor: { value: new Color('#f59e0b') },
        uColorDim: { value: new Color('#d97706') },
      },
      vertexShader: `
        uniform float uScale;
        attribute float aSize;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (aSize * uScale) * (200.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform vec3 uColorDim;
        void main() {
          // Circular point shape
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          // Soft falloff
          float alpha = smoothstep(0.5, 0.0, d) * 0.35;
          vec3 col = mix(uColorDim, uColor, 1.0 - d * 1.5);
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
    });
  }, []);

  return (
    <points ref={pointsRef} material={shaderMaterial}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={PARTICLE_COUNT}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-aSize"
          count={PARTICLE_COUNT}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
    </points>
  );
}

export default function AtomicBackground({ isChatbotOpen }) {
  const reducedMotion = useReducedMotion();
  const wrapRef = useRef(null);
  // Normalized mouse position: -1 to 1 (center = 0)
  const mouseRef = useRef({ x: 0, y: 0 });
  const [inHero, setInHero] = useState(true);
  const [tabVisible, setTabVisible] = useState(!document.hidden);
  const [settling, setSettling] = useState(false);

  useEffect(() => {
    const onMouseMove = (e) => {
      mouseRef.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouseRef.current.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener('mousemove', onMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMouseMove);
  }, []);

  // Fade the canvas from 1 to 0.12 over the first screen of scroll
  useEffect(() => {
    const onScroll = () => {
      const progress = Math.min(window.scrollY / window.innerHeight, 1);
      if (wrapRef.current) {
        wrapRef.current.style.setProperty('--scroll-fade', (1 - progress * 0.88).toFixed(3));
      }
      setInHero(progress < 1);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  useEffect(() => {
    const onVisibility = () => setTabVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  // Keep rendering briefly after the chatbot closes so the sphere eases back
  useEffect(() => {
    if (isChatbotOpen) return undefined;
    setSettling(true);
    const timer = setTimeout(() => setSettling(false), 1500);
    return () => clearTimeout(timer);
  }, [isChatbotOpen]);

  const animate = !reducedMotion && tabVisible && (inHero || isChatbotOpen || settling);

  return (
    <div
      ref={wrapRef}
      className="particle-canvas"
      style={{ opacity: isChatbotOpen ? 1 : 'var(--scroll-fade, 1)' }}
      aria-hidden="true"
    >
      <Canvas camera={{ position: [0, 0, CAMERA_Z], fov: 45 }} dpr={[1, 1.5]} frameloop={animate ? 'always' : 'demand'}>
        <color attach="background" args={['#0a0f1a']} />
        <ParticleSphere mouseRef={mouseRef} isChatbotOpen={isChatbotOpen} animate={animate} />
      </Canvas>
    </div>
  );
}
