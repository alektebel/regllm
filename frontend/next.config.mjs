// In ECS the API runs as a sidecar on localhost:8000. For local Docker Compose
// dev, set INTERNAL_API_URL=http://fastapi:8000 in your environment.
const internalApiUrl = process.env.INTERNAL_API_URL || "http://localhost:8000";
// SAS compiler endpoints run on 8001 to avoid conflicts with existing API container
const sasApiUrl = process.env.SAS_API_URL || "http://localhost:8001";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      // SAS router must come first — matched before the generic /api/* rule
      {
        source: "/api/sas/:path*",
        destination: `${sasApiUrl}/sas/:path*`,
      },
      {
        source: "/api/:path*",
        destination: `${internalApiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
