# JavaScript / TypeScript analysis image: Node + global eslint, typescript and
# dependency-cruiser so linting, type-checking and import analysis work even when
# the project ships none of them. Project deps + local tool binaries are
# installed at run time (npm cache at /root/.npm).
FROM node:22-alpine
RUN npm install -g eslint typescript dependency-cruiser
