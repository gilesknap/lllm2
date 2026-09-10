// Runs inside the real Pi jail. Keep routing diagnostics in CI so failures
// distinguish a reachable private destination from a missing gateway route.
import { spawnSync } from "node:child_process";
export default function (pi) {
  pi.on("session_start", () => {
    const result = spawnSync("bash", ["-c", `
      ip -4 address show
      ip -4 route show table all
      for target in 10.255.255.254 172.31.255.254 192.168.255.254 100.127.255.254; do
        echo "Route probe: $target"
        ip -4 route get "$target" || true
      done
      gateway=$(ip -4 route show default | awk 'NR == 1 {print $3}')
      echo "Gateway probe: $gateway"
      ip -4 route get "$gateway"
    `], { encoding: "utf8" });
    process.stderr.write(result.stdout + result.stderr);
    process.exit(result.status ?? 1);
  });
}
