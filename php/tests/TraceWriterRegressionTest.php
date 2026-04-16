<?php

declare(strict_types=1);

namespace Hop\Fit\Tests;

use Hop\Fit\Advice;
use Hop\Fit\Reward;
use Hop\Fit\Trace;
use Hop\Fit\TraceWriter;
use PHPUnit\Framework\TestCase;

class TraceWriterRegressionTest extends TestCase
{
    // PR#11 regression: mkdir/file_put_contents failures must throw.
    //
    // Before fix: mkdir() and file_put_contents() return values were
    // unchecked. A write to an unwritable path would silently succeed
    // (returning the path string) without the file actually being created.
    // After fix: both operations are checked and throw RuntimeException
    // on failure.

    public function testWriteToUnwritablePathThrows(): void
    {
        $advice = new Advice(domain: 'test', steeringText: 'steer', confidence: 0.5);
        $reward = new Reward(score: 0.9, breakdown: ['accuracy' => 0.9]);
        $trace = new Trace(
            id: 'trace-1',
            sessionId: 'sess-1',
            timestamp: '2026-04-16T00:00:00Z',
            input: ['prompt' => 'test'],
            advice: $advice,
            frontier: ['model' => 'stub'],
            reward: $reward,
        );

        // /dev/null is not a directory; mkdir will fail
        $writer = new TraceWriter('/dev/null');
        $this->expectException(\RuntimeException::class);
        $writer->write($trace, 1);
    }
}
