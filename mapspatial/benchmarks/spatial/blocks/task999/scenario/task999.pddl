(define (problem BLOCKS-999-0) (:domain blocks)
(:objects
	block-a - block
	block-b - block
    ag1 - agent
    big - size
)
(:init
	(handempty ag1)
	(clear block-a)
	(clear block-b)
	(onground block-a)
	(onground block-b)
	(blocktype big block-a)
	(blocktype big block-b)
)
(:goal
	(and
	    (handempty ag1)
		(on block-b block-a)
        (blocktype big block-a)
        (blocktype big block-b)
        (clear block-b)
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

