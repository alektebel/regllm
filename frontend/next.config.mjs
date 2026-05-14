// At build time inside Docker, INTERNAL_API_URL isn't injected, so default
// to the Docker service name. For local dev (no Docker) set INTERNAL_API_URL
// in frontend/.env.local to http://localhost:8000
const internalApiUrl = process.env.INTERNAL_API_URL || "http://fastapi:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${internalApiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
