# Java (Gradle) analysis image: the official Gradle+JDK base + a Checkstyle CLI.
# Pinned to root with GRADLE_USER_HOME under /root so the mounted Gradle cache
# volume (/root/.gradle) is writable regardless of the base image's default user.
FROM gradle:8-jdk21
USER root
ENV GRADLE_USER_HOME=/root/.gradle
ARG CHECKSTYLE_VERSION=10.17.0
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fL -o /opt/checkstyle.jar \
      "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VERSION}/checkstyle-${CHECKSTYLE_VERSION}-all.jar"; \
    printf '#!/bin/sh\nexec java -jar /opt/checkstyle.jar "$@"\n' > /usr/local/bin/checkstyle; \
    chmod +x /usr/local/bin/checkstyle
