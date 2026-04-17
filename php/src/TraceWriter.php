<?php

declare(strict_types=1);

namespace HopTop\Fit;

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
            $created = mkdir($sessionDir, 0755, true);
            if (!$created && !is_dir($sessionDir)) {
                throw new \RuntimeException("Failed to create directory: {$sessionDir}");
            }
        }
        $path = "{$sessionDir}/step-" . str_pad((string) $step, 3, '0', STR_PAD_LEFT) . '.yaml';
        $result = file_put_contents(
            $path,
            Yaml::dump($trace->toArray(), 2, 4, Yaml::DUMP_OBJECT_AS_MAP),
        );
        if ($result === false) {
            throw new \RuntimeException("Failed to write trace file: {$path}");
        }
        return $path;
    }
}
