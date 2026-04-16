<?php

declare(strict_types=1);

namespace Hop\Fit;

use Symfony\Component\Yaml\Yaml;

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
        file_put_contents(
            $path,
            Yaml::dump($trace->toArray(), 2, 4, Yaml::DUMP_OBJECT_AS_MAP),
        );
        return $path;
    }
}
