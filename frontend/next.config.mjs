// In ECS the API runs as a sidecar on localhost:8000. For local Docker Compose
// dev, set INTERNAL_API_URL=http://fastapi:8000 in your environment.
const internalApiUrl = process.env.INTERNAL_API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  // Allow access from any host on the local network during development
  allowedDevOrigins: ["*"],
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
