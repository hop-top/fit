<?php

declare(strict_types=1);

namespace HopTop\Fit\Tests;

use HopTop\Fit\Advice;
use PHPUnit\Framework\TestCase;

class AdviceTest extends TestCase
{
    public function testDefaults(): void
    {
        $a = new Advice('tax', 'cite sources', 0.9);
        $this->assertSame('tax', $a->domain);
        $this->assertSame([], $a->constraints);
        $this->assertSame('1.0', $a->version);
    }
}
