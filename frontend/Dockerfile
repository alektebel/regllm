# Multi-stage build: install deps → compile Next.js standalone → tiny runtime.
# Final image runs `next start` via the standalone server.js on port 3010.

FROM node:20-alpine AS deps
WORKDIR /app
ENV CI=1 \
    NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; \
    elif [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f pnpm-lock.yaml ]; then npm i -g pnpm && pnpm i --frozen-lockfile; \
    else npm install --no-audit --no-fund; fi

FROM node:20-alpine AS builder
WORKDIR /app
# Bake the API URL into the build because Next.js evaluates `rewrites()`
# at build time. Override with `--build-arg INTERNAL_API_URL=...`.
ARG INTERNAL_API_URL=http://api:8000
ENV NEXT_TELEMETRY_DISABLED=1 \
    INTERNAL_API_URL=$INTERNAL_API_URL
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3010 \
    HOSTNAME=0.0.0.0

# Non-root user
RUN addgroup -S app && adduser -S app -G app
USER app

COPY --from=builder --chown=app:app /app/public ./public
COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static

EXPOSE 3010
CMD ["node", "server.js"]
