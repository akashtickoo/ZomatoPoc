# The web front end as a container image (docs/deployment-plan.md, "Docker"): built with
# Node, then served as static files by nginx, running as an unprivileged user on port 8080.
#
#   docker build -t curated-cravings-web --build-arg VITE_API_BASE_URL=http://localhost:8000 web
#   docker run --rm -p 8080:8080 curated-cravings-web
#
# VITE_API_BASE_URL is compiled into the page, so it is public and fixed per image: it must
# be the API's address as the visitor's browser sees it, and the API must list this page's
# origin in CORS_ORIGINS.

# The Node that package.json's "engines" asks for, as on Vercel.
FROM node:24.22.2-alpine AS build
WORKDIR /web

# Dependencies first, exactly as locked, so a code change doesn't reinstall them.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
