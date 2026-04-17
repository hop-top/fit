<?php

declare(strict_types=1);

namespace HopTop\Fit\Tests;

use HopTop\Fit\CompositeScorer;
use HopTop\Fit\DimensionScorer;
use HopTop\Fit\Reward;
use HopTop\Fit\RewardScorerInterface;
use PHPUnit\Framework\TestCase;

class FixedScorer implements RewardScorerInterface
{
    public function __construct(private float $score, private array $breakdown = []) {}

    public function score(string $output, array $context): Reward
    {
        return new Reward($this->score, $this->breakdown);
    }
}

class CompositeScorerRegressionTest extends TestCase
{
    /**
     * Regression: CompositeScorer with empty scorers must not trigger
     * division-by-zero (1.0 / count([]) = 1.0 / 0).
     */
    public function testEmptyScorersNoDivisionByZero(): void
    {
        $scorer = new CompositeScorer([]);
        $result = $scorer->score('test', []);

        $this->assertEquals(0.0, $result->score);
        $this->assertSame([], $result->breakdown);
    }

    /**
     * Regression: default weights must be [] when scorers is empty,
     * not array_fill(0, 0, INF).
     */
    public function testEmptyScorersWeightsAreEmpty(): void
    {
        $scorer = new CompositeScorer([]);
        $result = $scorer->score('test', []);
        // Score of 0.0 confirms totalWeight was 0, not INF
        $this->assertEquals(0.0, $result->score);
    }

    /**
     * Regression: CompositeScorer must merge breakdowns from all scorers.
     */
    public function testMergesBreakdownsFromAllScorers(): void
    {
        $accuracy = new FixedScorer(0.9, ['accuracy' => 0.9]);
        $safety = new FixedScorer(0.7, ['safety' => 0.7]);

        $scorer = new CompositeScorer([$accuracy, $safety], [0.6, 0.4]);
        $result = $scorer->score('test', []);

        // Weighted: 0.9*0.6 + 0.7*0.4 = 0.54 + 0.28 = 0.82
        $this->assertEqualsWithDelta(0.82, $result->score, 0.001);
        // Before fix: only first scorer's breakdown was kept
        $this->assertArrayHasKey('accuracy', $result->breakdown);
        $this->assertArrayHasKey('safety', $result->breakdown);
    }

    /**
     * Regression: CompositeScorer must reject weights/scorers length mismatch.
     *
     * Before fix: score() silently used zip-like behavior, ignoring
     * mismatched weights or producing undefined offset errors.
     */
    public function testRejectsLengthMismatch(): void
    {
        $scorers = [
            new DimensionScorer('a'),
            new DimensionScorer('b'),
        ];

        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('mismatch');

        new CompositeScorer($scorers, [0.5, 0.3, 0.2]);
    }

    /**
     * Regression: weights shorter than scorers must also be rejected.
     */
    public function testRejectsShortWeights(): void
    {
        $scorers = [
            new DimensionScorer('a'),
            new DimensionScorer('b'),
            new DimensionScorer('c'),
        ];

        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('mismatch');

        new CompositeScorer($scorers, [0.5, 0.3]);
    }
}
