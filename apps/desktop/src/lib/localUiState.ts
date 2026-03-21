const MAX_RECENT_VALUES = 6;

function storageAvailable(): boolean {
  return Boolean(
    typeof window !== "undefined" &&
      window.localStorage &&
      typeof window.localStorage.getItem === "function" &&
      typeof window.localStorage.setItem === "function",
  );
}

export function loadRecentValues(key: string): string[] {
  if (!storageAvailable()) {
    return [];
  }
  const saved = window.localStorage.getItem(key);
  if (!saved) {
    return [];
  }
  try {
    const parsed = JSON.parse(saved) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((value): value is string => typeof value === "string");
  } catch {
    window.localStorage.removeItem(key);
    return [];
  }
}

export function rememberRecentValue(
  key: string,
  values: string[],
  nextValue: string,
  maxValues = MAX_RECENT_VALUES,
): string[] {
  const normalized = nextValue.trim();
  if (!normalized) {
    return values;
  }
  const nextValues = [normalized, ...values.filter((value) => value !== normalized)].slice(
    0,
    maxValues,
  );
  if (storageAvailable()) {
    window.localStorage.setItem(key, JSON.stringify(nextValues));
  }
  return nextValues;
}

export function persistRecentValues(key: string, values: string[]): void {
  if (!storageAvailable()) {
    return;
  }
  window.localStorage.setItem(key, JSON.stringify(values));
}

export async function copyText(text: string): Promise<boolean> {
  if (!text || typeof navigator === "undefined" || !navigator.clipboard) {
    return false;
  }
  await navigator.clipboard.writeText(text);
  return true;
}
