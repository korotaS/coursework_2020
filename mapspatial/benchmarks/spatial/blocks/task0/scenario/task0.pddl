(define (problem BLOCKS-1-0) (:domain blocks)
(:objects
	block-a - block
	block-c - block
	block-b - block
    ag1 - agent
    big - size
)
(:init
	(handempty ag1)
	(clear block-a)
	(clear block-b)
	(clear block-c)
	(onground block-a)
	(onground block-b)
	(onground block-c)
	(blocktype big block-a)
	(blocktype big block-b)
	(blocktype big block-c)
)
(:goal
	(and
	    (handempty ag1)
		(on block-b block-a)
		(on block-c block-b)
        (blocktype big block-a)
        (blocktype big block-b)
        (blocktype big block-c)
        (clear block-c)
        (onground block-a)
	)
)

(:constraints
    (and

        (and (always (forall (?x - block)
            (implies (blocktype big ?x)(holding ag1 ?x))))
        )

    )
)
)

