<?php

declare(strict_types=1);

namespace Hop\Fit;

/**
 * Composite scorer combining multiple scorers with configurable weights.
 *
 * final_score = weighted_sum(scorer_i.score, weight_i) / total_weight
 */
class CompositeScorer implements RewardScorerInterface
{
    /** @var RewardScorerInterface[] */
    private readonly array $scorers;
    /** @var float[] */
    private readonly array $weights;

    /**
     * @param RewardScorerInterface[] $scorers
     * @param float[] $weights Equal weights if empty.
     */
    public function __construct(array $scorers, array $weights = [])
    {
        $this->scorers = $scorers;
        $this->weights = empty($weights)
            ? (count($scorers) === 0
                ? []
                : array_fill(0, count($scorers), 1.0 / count($scorers)))
            : $weights;

        if (!empty($weights) && count($weights) !== count($scorers)) {
            throw new \InvalidArgumentException(sprintf(
                'weights/scorers length mismatch: %d scorers but %d weights',
                count($scorers),
                count($weights),
            ));
        }
    }

    public function score(string $output, array $context): Reward
    {
        $rewards = array_map(
            fn(RewardScorerInterface $s) => $s->score($output, $context),
            $this->scorers,
        );

        $totalWeight = array_sum($this->weights);
        if ($totalWeight == 0.0) {
            return new Reward(0.0, []);
        }

        $mergedBreakdown = [];
        foreach ($rewards as $reward) {
            $mergedBreakdown = array_merge($mergedBreakdown, $reward->breakdown);
        }

        // If any child score is null, propagate null (failure semantics per reward-schema-v1)
        foreach ($rewards as $reward) {
            if ($reward->score === null) {
                return new Reward(
                    score: null,
                    breakdown: $mergedBreakdown,
                    metadata: ['scorers' => count($rewards), 'error' => 'child_score_is_null'],
                );
            }
        }

        $combined = 0.0;
        foreach ($rewards as $i => $reward) {
            $combined += $reward->score * $this->weights[$i];
        }

        return new Reward(
            score: $combined / $totalWeight,
            breakdown: $mergedBreakdown,
            metadata: ['scorers' => count($rewards)],
        );
    }

    /**
     * Convenience: create from dimension names with equal weights.
     *
     * @param string[] $dimensions
     */
    public static function fromDimensions(array $dimensions): self
    {
        $scorers = array_map(
            fn(string $d) => new DimensionScorer($d),
            $dimensions,
        );
        return new self($scorers);
    }
}
