/**
 * ComputeBackend — conceptual interface (Dashboard Spec Ch. 71.8).
 *
 * Every concrete backend (Local, Colab, CollegeGPU, Demo) must expose
 * the same high-level operations so the dashboard/visualization layer
 * never has to know which backend is actually running the experiment.
 *
 * These classes are interface stubs behind the /status health check.
 * The working backends live in core/compute (Python); see
 * docs/COMPUTE_BACKENDS.md.
 */
export class ComputeBackend {
  /** @returns {Promise<{name: string, available: boolean}>} */
  async getStatus() {
    throw new Error("getStatus() not implemented");
  }

  /** @param {string} experimentId */
  async loadExperiment(experimentId) {
    throw new Error("loadExperiment() not implemented");
  }

  /** @param {string} experimentId @param {string} clientId */
  async getClientDistribution(experimentId, clientId) {
    throw new Error("getClientDistribution() not implemented");
  }

  /** @param {string} experimentId */
  async getFlHistory(experimentId) {
    throw new Error("getFlHistory() not implemented");
  }

  /** @param {string} experimentId @param {string} clientId */
  async startUnlearning(experimentId, clientId) {
    throw new Error("startUnlearning() not implemented");
  }

  /** @param {string} experimentId */
  async getUnlearningStatus(experimentId) {
    throw new Error("getUnlearningStatus() not implemented");
  }

  /** @param {string} experimentId */
  async getResults(experimentId) {
    throw new Error("getResults() not implemented");
  }
}
