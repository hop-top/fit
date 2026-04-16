import yaml from "js-yaml";

export function dump(obj: unknown): string {
  return yaml.dump(obj, { lineWidth: -1, noRefs: true });
}

export function load(text: string): unknown {
  return yaml.load(text);
}
