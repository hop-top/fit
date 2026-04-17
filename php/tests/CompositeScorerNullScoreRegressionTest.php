<?php

declare(strict_types=1);

namespace HopTop\Fit\Tests;

use HopTop\Fit\CompositeScorer;
use HopTop\Fit\Reward;
use HopTop\Fit\RewardScorerInterface;
use PHPUnit\Framework\TestCase;

class NullScorer implements RewardScorerInterface
{
    public function score(string $output, array $context): Reward
    {
        return new Reward(score: null, breakdown: ['failed' => 0.0]);
    }
}

class FixedValueScorer implements RewardScorerInterface
{
    public function __construct(private float $score, private array $breakdown = []) {}

    public function score(string $output, array $context): Reward
    {
        return new Reward($this->score, $this->breakdown);
    }
}

class CompositeScorerNullScoreRegressionTest extends TestCase
{
    /**
     * PR#16: CompositeScorer must return score=null when any child
     * scorer returns null, instead of crashing on null * float.
     */
    public function testReturnsNullWhenOneChildIsNull(): void
    {
        $good = new FixedValueScorer(0.9, ['accuracy' => 0.9]);
        $bad = new NullScorer();
        $scorer = new CompositeScorer([$good, $bad], [0.5, 0.5]);

        $result = $scorer->score('test', []);

        $this->assertNull($result->score);
        $this->assertSame('child_score_is_null', $result->metadata['error'] ?? null);
    }

    /**
     * PR#16: Breakdown must be merged even when propagating null score.
     */
    public function testNullPropagatesBreakdown(): void
    {
        $good = new FixedValueScorer(0.8, ['accuracy' => 0.8]);
        $bad = new NullScorer();
        $scorer = new CompositeScorer([$good, $bad], [0.7, 0.3]);

        $result = $scorer->score('test', []);

        $this->assertNull($result->score);
        $this->assertArrayHasKey('accuracy', $result->breakdown);
        $this->assertArrayHasKey('failed', $result->breakdown);
    }

    /**
     * PR#16: All children null must still return null score.
     */
    public function testAllChildrenNull(): void
    {
        $scorer = new CompositeScorer([new NullScorer(), new NullScorer()]);

        $result = $scorer->score('test', []);

        $this->assertNull($result->score);
        $this->assertSame('child_score_is_null', $result->metadata['error'] ?? null);
    }

    /**
     * PR#19: zero-weight composite with null child must return
     * null, not 0.0. Null check must happen before totalWeight
     * short-circuit.
     */
    public function testZeroWeightWithNullChildReturnsNull(): void
    {
        $nullScorer = new NullScorer();
        $okScorer = new FixedValueScorer(0.5, ['ok' => 0.5]);
        $scorer = new CompositeScorer([$nullScorer, $okScorer], [0.0, 0.0]);

        $result = $scorer->score('test', []);

        $this->assertNull($result->score);
        $this->assertSame('child_score_is_null', $result->metadata['error'] ?? null);
    }

    /**
     * PR#16: No null scores must still compute weighted average.
     */
    public function testNoNullStillWorks(): void
    {
        $a = new FixedValueScorer(1.0, ['a' => 1.0]);
        $b = new FixedValueScorer(0.0, ['b' => 0.0]);
        $scorer = new CompositeScorer([$a, $b], [0.5, 0.5]);

        $result = $scorer->score('test', []);

        $this->assertEqualsWithDelta(0.5, $result->score, 0.001);
        $this->assertArrayNotHasKey('error', $result->metadata);
    }
}
