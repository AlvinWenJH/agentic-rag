import fs from "fs";
import path from "path";

const rootEnv = path.resolve(process.cwd(), "..", "frontend.env");
const localEnv = path.resolve(process.cwd(), ".env.local");

try {
  if (fs.existsSync(rootEnv)) {
    fs.copyFileSync(rootEnv, localEnv);
    console.log(`[dev] Loaded env from ${rootEnv} -> ${localEnv}`);
  } else {
    console.warn(`[dev] No frontend.env found at ${rootEnv}. Using existing .env.local if present.`);
  }
} catch (err) {
  console.error(`[dev] Failed to prepare .env.local:`, err);
}