<?php

declare(strict_types=1);

namespace Hop\Fit;

class TraceWriter
{
    public function __construct(
        private string $outputDir,
    ) {}

    public function write(Trace $trace, int $step = 1): string
    {
        $sessionDir = "{$this->outputDir}/{$trace->sessionId}";
        if (!is_dir($sessionDir)) {
            mkdir($sessionDir, 0755, true);
        }
        $path = "{$sessionDir}/step-" . str_pad((string) $step, 3, '0', STR_PAD_LEFT) . '.yaml';
        file_put_contents($path, yaml_emit($trace->toArray() ?? []));
        return $path;
    }
}
