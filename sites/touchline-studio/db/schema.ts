import {sqliteTable,text} from 'drizzle-orm/sqlite-core';
export const subscribers=sqliteTable('subscribers',{email:text('email').primaryKey(),createdAt:text('created_at').notNull(),consent:text('consent').notNull(),removalToken:text('removal_token').notNull().unique()});
export const forecastCache=sqliteTable('forecast_cache',{key:text('key').primaryKey(),payload:text('payload').notNull(),updatedAt:text('updated_at').notNull()});
