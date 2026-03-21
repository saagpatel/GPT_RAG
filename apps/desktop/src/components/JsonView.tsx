export function JsonView({ value }: { value: unknown }) {
  return (
    <pre className="json-view">
      <code>{JSON.stringify(value, null, 2)}</code>
    </pre>
  );
}
