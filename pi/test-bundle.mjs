// Integration probe run inside the disposable outer container by smoke.sh.
import { spawn } from "node:child_process";
import { strict as assert } from "node:assert";

// Optional explicit binary/package-root arguments support a host compatibility
// probe with an isolated HOME; CI uses the actual image launcher by default.
const args = ["--mode", "rpc", "--no-session"];
if (process.argv[3]) {
  for (const name of ["pi-mcp-adapter", "pi-web-access", "pi-powerline"]) {
    args.push("--extension", `${process.argv[3]}/${name}`);
  }
}
const child = spawn(process.argv[2] || "pi", args, {
  env: { ...process.env, PI_OFFLINE: "1", CLAUDE_SANDBOX_PASS_ENV: "PI_OFFLINE" },
  stdio: ["pipe", "pipe", "pipe"],
});
let stdout = "", stderr = "", complete = false, answered = false;
const timer = setTimeout(() => {
  process.stderr.write(stdout + stderr);
  child.kill("SIGTERM");
  process.exitCode = 1;
}, 60000);
child.stdout.on("data", (chunk) => {
  stdout += chunk;
  for (const line of stdout.split("\n")) {
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    if (answered) return;
    if (message.type !== "response" || message.command !== "get_commands") continue;
    answered = true;
    try {
      assert.equal(message.success, true);
      for (const name of ["mcp", "websearch", "powerline"]) {
        assert(message.data.commands.some((command) => command.name === name),
          `Bundled extension did not register /${name}`);
      }
      assert(!/failed to load extension|error loading extension/i.test(stdout + stderr),
        stdout + stderr);
      complete = true;
    } catch (error) {
      process.stderr.write(String(error) + "\n" + stdout + stderr);
      process.exitCode = 1;
    }
    clearTimeout(timer);
    child.stdin.end();
    child.kill("SIGTERM");
    return;
  }
});
child.stderr.on("data", (chunk) => { stderr += chunk; });
child.on("error", (error) => {
  clearTimeout(timer);
  throw error;
});
child.on("close", () => {
  clearTimeout(timer);
  if (!complete) {
    process.stderr.write("Pi bundle did not start\n" + stdout + stderr);
    process.exitCode = 1;
  }
});
child.stdin.on("error", (error) => {
  if (!answered) { process.stderr.write(String(error)); process.exitCode = 1; }
});
child.stdin.write('{"id":"bundle-test","type":"get_commands"}\n');
