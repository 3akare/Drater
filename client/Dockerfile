# Build stage
FROM node:20.18.0 AS builder

WORKDIR /app

RUN npm install -g pnpm

COPY package*.json ./

RUN pnpm install

COPY . .

RUN pnpm run build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/out /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
