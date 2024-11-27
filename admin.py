import click
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base, Region, BuildingDetails, building_regions
from datetime import datetime
import json
from urllib.parse import urlparse

load_dotenv()



def get_session_for_environment(env: str):
    """Get database session for specified environment."""
    try:
        # Get appropriate URL based on environment
        url = os.getenv("PROD_DATABASE_URL" if env == "prod" else "DATABASE_URL")
        if not url:
            raise ValueError(f"{'PROD_DATABASE_URL' if env == 'prod' else 'DATABASE_URL'} environment variable is not set")
        
        # Parse and validate URL
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
            
        try:
            result = urlparse(url)
            if not all([result.scheme, result.hostname, result.username, result.password]):
                raise ValueError("Invalid database URL format")
        except Exception as e:
            raise ValueError(f"Failed to parse database URL: {str(e)}")
        
        # Create engine with connection pooling
        engine = create_engine(url, 
            pool_recycle=1800,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        
        # Test connection
        engine.connect()
        
        return sessionmaker(bind=engine)()
        
    except Exception as e:
        click.echo(f"Error connecting to database: {str(e)}", err=True)
        raise

@click.group()
def cli():
    """Admin CLI for managing the virtual campus."""
    pass

@cli.command()
@click.option('--id', required=True, help='Region ID')
@click.option('--name', required=True, help='Region name')
@click.option('--color', required=True, help='Region color (hex)')
@click.option('--center', required=True, type=str, help='Region center coordinates as JSON array [x, y]')
@click.option('--description', help='Region description')
@click.option('--env', default='prod', help='Environment to use (prod or dev)')
def add_region(id: str, name: str, color: str, center: str, description: str = None, env: str = 'prod'):
    """Add a hardcoded region."""
    try:
        center_coords = json.loads(center)
        if not isinstance(center_coords, list) or len(center_coords) != 2:
            raise ValueError("Center must be a JSON array with 2 coordinates")
            
        session = get_session_for_environment(env)
        region = Region(
            id=id,
            name=name,
            color=color,
            center=center_coords,
            is_hardcoded=True,
            description=description
        )
        session.add(region)
        session.commit()
        click.echo(f"Added region: {name}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
    finally:
        session.close()

@cli.command()
@click.option('--building-id', required=True, help='Building ID')
@click.option('--region-id', required=True, help='Region ID')
@click.option('--env', default='prod', help='Environment to use (prod or dev)')
def assign_building(building_id: str, region_id: str, env: str = 'prod'):
    """Assign a building to a hardcoded region."""
    try:
        session = get_session_for_environment(env)
        
        # Verify building and region exist
        building = session.query(BuildingDetails).filter_by(building_id=building_id).first()
        region = session.query(Region).filter_by(id=region_id).first()
        
        if not building:
            raise ValueError(f"Building {building_id} not found")
        if not region:
            raise ValueError(f"Region {region_id} not found")
            
        # Add relationship
        if region not in building.regions:
            building.regions.append(region)
            session.commit()
            click.echo(f"Assigned building {building_id} to region {region_id}")
        else:
            click.echo("Building already assigned to this region")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
    finally:
        session.close()

@cli.command()
@click.option('--env', default='prod', help='Environment to use (prod or dev)')
def list_regions(env: str = 'prod'):
    """List all regions."""
    try:
        session = get_session_for_environment(env)
        regions = session.query(Region).all()
        
        for region in regions:
            click.echo(f"\nRegion: {region.name} ({region.id})")
            click.echo(f"Type: {'Hardcoded' if region.is_hardcoded else 'Dynamic'}")
            click.echo(f"Buildings: {len(region.buildings)}")
            click.echo(f"Center: {region.center}")
            if region.description:
                click.echo(f"Description: {region.description}")
                
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
    finally:
        session.close()

@cli.command()
@click.option('--region-id', required=True, help='Region ID')
@click.option('--env', default='prod', help='Environment to use (prod or dev)')
def list_buildings(region_id: str, env: str = 'prod'):
    """List all buildings in a region."""
    try:
        session = get_session_for_environment(env)
        region = session.query(Region).filter_by(id=region_id).first()
        
        if not region:
            raise ValueError(f"Region {region_id} not found")
            
        click.echo(f"\nBuildings in {region.name}:")
        for building in region.buildings:
            click.echo(f"- {building.building_id}: {building.full_building_type}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
    finally:
        session.close()

@cli.command()
@click.option('--env', default='prod', help='Environment to use (prod or dev)')
def test_connection(env: str = 'prod'):
    """Test database connection."""
    try:
        session = get_session_for_environment(env)
        # Try a simple query
        result = session.execute("SELECT 1").scalar()
        click.echo(f"Successfully connected to {env} database")
        session.close()
    except Exception as e:
        click.echo(f"Error connecting to database: {str(e)}", err=True)

if __name__ == '__main__':
    cli() 