import { DemoBackend } from "./DemoBackend.js";
import { LocalBackend } from "./LocalBackend.js";
import { ColabBackend } from "./ColabBackend.js";
import { CollegeGPUBackend } from "./CollegeGPUBackend.js";

/**
 * ComputeManager — picks a backend by name (env: COMPUTE_BACKEND).
 * "auto" priority per Ch. 71.16: college-gpu -> colab -> local -> demo.
 * Only "demo" reports available here; the others are stubs. The working
 * College GPU / Colab / Local / Demo backends are in core/compute and are
 * reached through /api/compute.
 */
const BACKENDS = {
  demo: DemoBackend,
  local: LocalBackend,
  colab: ColabBackend,
  "college-gpu": CollegeGPUBackend,
};

export function getComputeBackend(name = process.env.COMPUTE_BACKEND || "demo") {
  const BackendClass = BACKENDS[name] || DemoBackend;
  return new BackendClass();
}
