# Lists all users with basic details
docker-compose exec db psql -U postgres -d image_api -c "SELECT id, username, email, team_id, is_admin, created_at FROM users ORDER BY created_at DESC;"

# Count how many users exist per team and show the team name
docker-compose exec db psql -U postgres -d image_api -c "
SELECT t.name as team_name, COUNT(u.id) as user_count 
FROM users u 
JOIN teams t ON u.team_id = t.id 
GROUP BY t.name 
ORDER BY user_count DESC;"

# Check users from a specific team (replace the team name)
docker-compose exec db psql -U postgres -d image_api -c "
SELECT u.username, u.email, u.created_at 
FROM users u 
JOIN teams t ON u.team_id = t.id 
WHERE t.name = 'AI Research Team' 
ORDER BY u.created_at DESC;"

# Verify users created in the last 24 hours
docker-compose exec db psql -U postgres -d image_api -c "
SELECT username, email, team_id, created_at 
FROM users 
WHERE created_at > NOW() - INTERVAL '24 hours' 
ORDER BY created_at DESC;"

