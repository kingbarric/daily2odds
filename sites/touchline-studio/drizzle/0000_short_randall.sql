CREATE TABLE `subscribers` (
	`email` text PRIMARY KEY NOT NULL,
	`created_at` text NOT NULL,
	`consent` text NOT NULL,
	`removal_token` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `subscribers_removal_token_unique` ON `subscribers` (`removal_token`);