const API_BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.message || message;
    } catch {
      // ignore parsing error
    }
    throw new Error(message);
  }

  return response.json();
}

export function searchIdentity(payload) {
  return request("/api/search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function verifyIdentity(payload) {
  return request("/api/verify", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSystemOverview() {
  return request("/api/system/overview");
}

export function getTaskDetail(taskId) {
  return request(`/api/tasks/${taskId}`);
}

export function getUserSuggestions(platform, query, limit = 8) {
  const params = new URLSearchParams({
    platform,
    q: query,
    limit: String(limit),
  });
  return request(`/api/users/suggest?${params.toString()}`);
}
