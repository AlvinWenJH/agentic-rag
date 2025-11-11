import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ensure Dockerfile can copy from .next/standalone
  output: "standalone",
  reactCompiler: true,
};

export default nextConfig;
