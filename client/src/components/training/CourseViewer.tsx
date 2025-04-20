import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link, useParams, useNavigate } from 'react-router-dom';
import Markdown from 'markdown-to-jsx';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CourseMetadata {
  id: string;
  title: string;
  category: string;
  description: string;
  file_path: string;
}

interface CourseContent {
  id: string;
  title: string;
  category: string;
  description: string;
  content: string;
  content_html: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Custom code renderer for markdown-to-jsx
const CodeBlock = ({ className, children }: { className?: string; children: string }) => {
  // Extract language from className (format is: "language-xxx")
  const language = className ? className.replace('language-', '') : 'javascript';
  
  return (
    <SyntaxHighlighter language={language} style={tomorrow}>
      {children}
    </SyntaxHighlighter>
  );
};

export const CourseList: React.FC = () => {
  const [courses, setCourses] = useState<CourseMetadata[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>('all');

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/courses`);
        setCourses(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching courses:', err);
        setError('Failed to load courses. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchCourses();
  }, []);

  const filteredCourses = activeCategory === 'all' 
    ? courses 
    : courses.filter(course => course.category === activeCategory);

  const handleCategoryChange = (category: string) => {
    setActiveCategory(category);
  };

  if (loading) {
    return <div className="p-4 text-center">Loading courses...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-600 text-center">{error}</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Security AI Courses</h1>
      
      <div className="flex mb-6 border-b">
        <button 
          className={`px-4 py-2 ${activeCategory === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => handleCategoryChange('all')}
        >
          All Courses
        </button>
        <button 
          className={`px-4 py-2 ${activeCategory === 'basics' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => handleCategoryChange('basics')}
        >
          Basics
        </button>
        <button 
          className={`px-4 py-2 ${activeCategory === 'advanced' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => handleCategoryChange('advanced')}
        >
          Advanced
        </button>
        <button 
          className={`px-4 py-2 ${activeCategory === 'expert' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => handleCategoryChange('expert')}
        >
          Expert
        </button>
      </div>
      
      {filteredCourses.length === 0 ? (
        <div className="text-center p-4">No courses found</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCourses.map(course => (
            <div key={`${course.category}-${course.id}`} className="border rounded-lg overflow-hidden shadow-lg">
              <div className={`${categoryColor(course.category)} h-2`}></div>
              <div className="p-4">
                <h2 className="text-xl font-semibold mb-2">{course.title}</h2>
                <div className="text-sm text-gray-600 mb-2">
                  Category: <span className="font-medium">{capitalizeFirstLetter(course.category)}</span>
                </div>
                <p className="text-gray-700 mb-4 line-clamp-3">{course.description}</p>
                <Link 
                  to={`/courses/${course.category}/${course.id}`}
                  className="inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  View Course
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export const CourseViewer: React.FC = () => {
  const { category, courseId } = useParams<{ category: string; courseId: string }>();
  const [course, setCourse] = useState<CourseContent | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCourseContent = async () => {
      if (!category || !courseId) {
        setError('Invalid course URL');
        return;
      }

      try {
        setLoading(true);
        const response = await axios.get(
          `${API_BASE_URL}/courses/${category}/${courseId}`
        );
        setCourse(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching course content:', err);
        setError('Failed to load course content. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchCourseContent();
  }, [category, courseId]);

  const handleBack = () => {
    navigate('/courses');
  };

  if (loading) {
    return <div className="p-4 text-center">Loading course content...</div>;
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="text-red-600 mb-4">{error}</div>
        <button
          onClick={handleBack}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Back to Courses
        </button>
      </div>
    );
  }

  if (!course) {
    return (
      <div className="p-4">
        <div className="text-red-600 mb-4">Course not found</div>
        <button
          onClick={handleBack}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Back to Courses
        </button>
      </div>
    );
  }

  // Options for markdown-to-jsx
  const markdownOptions = {
    overrides: {
      code: {
        component: CodeBlock
      },
      pre: {
        component: ({ children, ...props }: { children: React.ReactNode }) => <div {...props}>{children}</div>
      },
      h1: {
        component: ({ children }: { children: React.ReactNode }) => (
          <h1 className="text-3xl font-bold mt-6 mb-3">{children}</h1>
        )
      },
      h2: {
        component: ({ children }: { children: React.ReactNode }) => (
          <h2 className="text-2xl font-semibold mt-5 mb-3">{children}</h2>
        )
      },
      h3: {
        component: ({ children }: { children: React.ReactNode }) => (
          <h3 className="text-xl font-medium mt-4 mb-2">{children}</h3>
        )
      },
      p: {
        component: ({ children }: { children: React.ReactNode }) => (
          <p className="my-3">{children}</p>
        )
      },
      ul: {
        component: ({ children }: { children: React.ReactNode }) => (
          <ul className="list-disc ml-6 my-3">{children}</ul>
        )
      },
      ol: {
        component: ({ children }: { children: React.ReactNode }) => (
          <ol className="list-decimal ml-6 my-3">{children}</ol>
        )
      },
      a: {
        component: ({ children, href }: { children: React.ReactNode, href?: string }) => (
          <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        )
      }
    }
  };

  return (
    <div className="container mx-auto p-4">
      <button
        onClick={handleBack}
        className="mb-4 px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded inline-flex items-center"
      >
        <span className="mr-2">←</span> Back to Courses
      </button>
      
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className={`${categoryColor(course.category)} h-2`}></div>
        <div className="p-6">
          <h1 className="text-3xl font-bold mb-2">{course.title}</h1>
          <div className="text-sm text-gray-600 mb-6">
            Category: <span className="font-medium">{capitalizeFirstLetter(course.category)}</span>
          </div>
          
          <div className="prose prose-lg max-w-none">
            <Markdown options={markdownOptions}>
              {course.content}
            </Markdown>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper functions
function categoryColor(category: string): string {
  switch (category) {
    case 'basics':
      return 'bg-green-500';
    case 'advanced':
      return 'bg-blue-500';
    case 'expert':
      return 'bg-purple-500';
    default:
      return 'bg-gray-500';
  }
}

function capitalizeFirstLetter(string: string): string {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

export default CourseViewer; 