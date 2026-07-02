FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npx ng build --configuration production

FROM nginx:alpine
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/dqc-app/browser /usr/share/nginx/html

# API_UPSTREAM is resolved by nginx's envsubst-on-start:
#   docker-compose: API_UPSTREAM=regllm-api:8000
#   ECS Fargate:    API_UPSTREAM=localhost:8000
ENV API_UPSTREAM=regllm-api:8000

EXPOSE 80
