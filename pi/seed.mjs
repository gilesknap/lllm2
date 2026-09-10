// Create defaults once. Existing settings, including malformed JSON, belong
// to the user and must never be merged, replaced, or repaired by the image.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export async function seed(directory, defaults) {
  await mkdir(directory, { recursive: true });
  try {
    await writeFile(join(directory, "settings.json"), defaults, {
      flag: "wx", mode: 0o600,
    });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
  }
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  await seed(join(process.env.HOME, ".pi", "agent"),
    await readFile(new URL("./defaults.json", import.meta.url)));
}
