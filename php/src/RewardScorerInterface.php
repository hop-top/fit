<?php

declare(strict_types=1);

namespace HopTop\Fit;

interface RewardScorerInterface
{
    public function score(string $output, array $context): Reward;
}
