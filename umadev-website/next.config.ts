import type { NextConfig } from "next";

const isGithubPages = process.env.GITHUB_PAGES === "true";
const githubPagesRepo = process.env.GITHUB_PAGES_REPO ?? "umadev";
const customDomain = process.env.GITHUB_PAGES_DOMAIN;
const basePath = isGithubPages && !customDomain ? `/${githubPagesRepo}` : "";

const nextConfig: NextConfig = {
  output: isGithubPages ? "export" : undefined,
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  images: {
    unoptimized: true,
  },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
