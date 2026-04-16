export function dump(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

export function load(text: string): unknown {
  return JSON.parse(text);
}
