export default function RuntimeConfigScript() {
  const cfg = {
    BACKEND_URL:
      process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || "",
  };

  const json = JSON.stringify(cfg);

  return (
    <script
      dangerouslySetInnerHTML={{
        __html: `window.__RUNTIME_CONFIG__=${json};`,
      }}
    />
  );
}