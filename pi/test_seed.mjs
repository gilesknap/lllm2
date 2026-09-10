import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { seed } from "./seed.mjs";

test("fresh settings are seeded; existing and malformed settings survive", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-seed-"));
  try {
    const file = join(directory, "settings.json");
    await seed(directory, '{"theme":"dark"}\n');
    assert.equal(await readFile(file, "utf8"), '{"theme":"dark"}\n');
    for (const contents of ['{"theme":"light","packages":["custom"]}', 'not JSON']) {
      await writeFile(file, contents);
      await seed(directory, '{}');
      assert.equal(await readFile(file, "utf8"), contents);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
